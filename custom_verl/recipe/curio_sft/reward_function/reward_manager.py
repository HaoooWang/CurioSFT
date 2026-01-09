
import threading
import torch
import requests
from math_verify import parse, verify
from verl import DataProto
from tqdm import tqdm
from typing import Any,Dict,List
from recipe.curio_sft.reward_function.oat_math_grader  import boxed_reward_fn as oat_evaluate 

def labeling_responses(responses: list[str], golden_answer: str,timeout=False):
    if not timeout:
        predict_answers = [parse(responses[i],parsing_timeout=None) for i in range(len(responses))]
        golden_answer = parse("$" + golden_answer + "$",parsing_timeout=None)
        labels = [verify(golden_answer, predict_answers[i],timeout_seconds=None) for i in range(len(responses))]
    else:
        predict_answers = list(map(parse, responses))
        golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
        labels = list(map(verify, golden_answers, predict_answers))
    return labels,predict_answers[0]


def process_item_external(args):
    i, sequences_str, ground_truth, reward_impl_version, oat_grader, timeout, valid_response_length,data_source,extra_info = args
    
    final_score = 0
    think_format = True if reward_impl_version else False
    if think_format:
        if  '</think>' in sequences_str:
            model_solution = sequences_str.split('</think>')[1]
        else:
            final_score = 0.0
            model_solution = ""
    else:
        model_solution = sequences_str
    
    labels,extracted_answers = labeling_responses([model_solution], ground_truth,timeout=timeout)
    
    if labels[0]:
        final_score = 1

    if oat_grader:
        oat_score = oat_evaluate(sequences_str, ground_truth, fast=False)[1]   
        final_score = max(float(oat_score), float(final_score))         

    return {'i': i, 'score': final_score, 'valid_response_length': valid_response_length,'extracted_answers':extracted_answers}


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version=1,validation=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.parallel = True
        self.reward_impl_version = reward_impl_version
        self.oat_grader = True

    def _call_external_api(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            payload = {'items': items}
            response = requests.post(
                f"http://127.0.0.1:19876/process_batch",
                json=payload,
                timeout=240
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('results', [])
            else:
                print(f"API call failed with status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"API call exception: {e}")
            return []

    def __call__(self, data: DataProto, return_dict=None):
        """We will expand this function gradually based on the available datasets"""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        extracted_answers = [''] * reward_tensor.shape[0]

        decoded_sequences = []
        valid_response_lengths = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            
            think_format = True if self.reward_impl_version else False
            if think_format:
                sequences_str =  '<think>\n' + sequences_str

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            
            decoded_sequences.append(sequences_str)
            valid_response_lengths.append(valid_response_length)
            ground_truths.append(ground_truth)
            data_sources.append(data_item.non_tensor_batch['data_source'])
            extra_infos.append(data_item.non_tensor_batch['extra_info'])
        if not self.parallel:
            args = [(i, decoded_sequences[i], ground_truths[i], self.reward_impl_version, 
                    self.oat_grader, True, valid_response_lengths[i],data_sources[i],extra_infos[i]) for i in range(len(data))]
            results = [process_item_external(args[i]) for i in tqdm(range(len(args)), desc="Evaluate (Sequential)")]
        else:
            print('---------------------------------parallel request---------------------------------')
            api_items = []
            for i in range(len(decoded_sequences)):
                item = {
                    'i': i,
                    'sequences_str': decoded_sequences[i],
                    'ground_truth': ground_truths[i],
                    'reward_impl_version': self.reward_impl_version,
                    'oat_grader': self.oat_grader,
                    'valid_response_length': int(valid_response_lengths[i]),
                    'enable_timeout': True,
                    'data_source': data_sources[i],
                    'extra_info': extra_infos[i]
                }
                api_items.append(item)
            
            results = self._call_external_api(api_items)
            if len(results) == 0:
                print('##'*50)
                print("API call failed, fallback to local processing...")
                print('##'*50)
                args = [(i, decoded_sequences[i], ground_truths[i], self.reward_impl_version, 
                        self.oat_grader, True, valid_response_lengths[i],data_sources[i],extra_infos[i]) for i in range(len(data))]
                results = [process_item_external(args[i]) for i in tqdm(range(len(args)), desc="Evaluate (Sequential)")]
            else:
                results = results
            
        results.sort(key=lambda x: x['i'])
        
        for result in results:
            i = result['i']
            score = result['score']
            extracted_answers[i] = result['extracted_answers']
            valid_response_length = result['valid_response_length']
            reward_tensor[i, valid_response_length - 1] = score

        return {"reward_tensor": reward_tensor, "reward_extra_info": {"extracted_answers": extracted_answers}}
