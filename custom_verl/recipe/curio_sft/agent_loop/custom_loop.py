# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import aiohttp
import warnings
import random
from typing import Any
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import requests

from pandas.core.base import NoNewAttributesMixin
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from math_verify import parse, verify

from recipe.curio_sft.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "ERROR"))


@register("curiosft_agent")
class CurioSFTAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level CurioSFTAgentLoop initialization")
        
        for name in ("math_verify", "math_verify.parser", "math_verify.grader"):
            logging.getLogger(name).setLevel(logging.ERROR)

        # Initialize tools from config file
        cls.tokenizer = tokenizer

        cls.reward_impl_version = config.reward_model.reward_impl_version
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.session = None
        verify_workers = 8 
        cls._verify_executor = ThreadPoolExecutor(
            max_workers=verify_workers,
            thread_name_prefix="math_verify_"
        )
        logger.info(f"Created math verify thread pool with {verify_workers} workers")

    def __del__(self):
        if hasattr(self, 'session') and self.session is not None:
            try:
                if not self.session.closed:
                    asyncio.create_task(self.session.close())
            except Exception as e:
                logger.warning(f"Failed to close session in __del__: {e}")

    async def close(self):
        if hasattr(self, 'session') and self.session is not None and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Failed to close session: {e}")

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], trajectory: dict[str, Any], exploration_info: dict[str, Any], val_reward_fn = None) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        logprobs = []
        response_mask = []
        source_mask = []
        reward = []
        tag = ""
        seq_type = 0
        question_index = trajectory['sample_index']
        current_sampling_params = sampling_params.copy()
        current_sampling_params['stop_token_ids'] = [151643, 151645]
        num_external_turns = 0
        external_prob = 0
        if exploration_info['strategy'] == 'naive':
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=current_sampling_params
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            source_mask += [0] * len(response_ids)
            logprobs = [0.0] * len(response_ids)

        elif exploration_info['strategy'] == 'naive_with_reward_model':
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=current_sampling_params
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            source_mask += [0] * len(response_ids)
            logprobs = [0.0] * len(response_ids)
            with simple_timer("verify_response", metrics):
                response_text = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False))
                current_reward = await self.verify_response(response_text, exploration_info['ground_truth']) 
                reward += [current_reward] * len(response_ids)
            
        elif exploration_info['strategy'] == 'fixed_external':
            external_response = exploration_info['external_response']
            assert external_response is not None, "external_response is required"
            response_ids = await self.loop.run_in_executor(None, lambda: self.tokenizer.encode(external_response, add_special_tokens=False))
            response_ids = response_ids + [self.tokenizer.eos_token_id]
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            source_mask += [1] * len(response_ids)
            seq_type = 1
            logprobs = [0.0] * len(response_ids)
            external_prob = 1.0
            num_external_turns += 1
       
        else:
            raise ValueError(f"Invalid exploration strategy: {exploration_info['strategy']}")

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        if len(reward) == 0:
            reward = [0.0] * len(source_mask[:self.response_length])

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            logprobs = logprobs,
            num_turns=num_external_turns,
            metrics=metrics,
            source_mask=source_mask[:self.response_length],
            reward=reward[:self.response_length],
            tag=tag,
            seq_type=seq_type,
            question_index=question_index,
            external_prob=external_prob,
        )
        return output

    
   
    
    async def _call_verify_api(self, text: str, ground_truth: str) -> float:
        try:
            payload = {
                'text': text,
                'ground_truth': ground_truth,
                'reward_impl_version': self.reward_impl_version
            }
            response = requests.post(
                f"http://127.0.0.1:19876/verify_response",
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('score', 0.0)
            else:
                logger.error(f"API verify_response failed with status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"API verify_response failed with exception: {e}")
            return None

    async def verify_response(self, text, ground_truth, max_execution_time=5):
        try:
            api_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._call_verify_api_sync(text, ground_truth)
                ),
                timeout=max_execution_time
            )
            
            if api_result is not None:
                return api_result
            return 0
                
        except asyncio.TimeoutError:
            logger.error(f"verify_response timed out after {max_execution_time} seconds")
            return 0
        except Exception as e:
            logger.error(f"verify_response failed: {str(e)}")
            return 0

    def _call_verify_api_sync(self, text: str, ground_truth: str) -> float:
        try:
            payload = {
                'i': 0,  
                'sequences_str': text,
                'ground_truth': ground_truth,
                'reward_impl_version': self.reward_impl_version,
                'oat_grader': False,  
                'valid_response_length': 1,  
                'enable_timeout': False
            }
            
            response = requests.post(
                f"http://127.0.0.1:19876/process_single",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('error') is None:
                    return result.get('score', 0.0)
                else:
                    logger.error(f"API process_single failed with error: {result.get('error')}")
                    return 0
            else:
                logger.error(f"API process_single failed with status code: {response.status_code}")
                return 0
                
        except Exception as e:
            logger.error(f"API process_single failed with exception: {e}")
            return 0
