# run on 8xH100
# make sure your current working directory is the root of the project

set -x
ulimit -n 65535

export HYDRA_FULL_ERROR=1
export NO_PROXY="127.0.0.1,localhost"
export VLLM_USE_V1=1


PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${TRAIN_CKPT_PATH:-"${PROJECT_DIR}/models/Qwen2.5-Math-7B-Thinking"}"

CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"
TRAIN_FILE="$PROJECT_DIR/data/train.parquet"
VAL_FILE="$PROJECT_DIR/data/validation.parquet"

MAX_RESPONSE_LENGTH=8192
WANDB_PROJECT="CurioSFT"
EXP_NAME="CurioSFT_SFT_Qwen2.5-Math-7B"
CKPTS_DIR="$PROJECT_DIR/exp_results/${EXP_NAME}"

ray start --head
mkdir -p ${CKPTS_DIR}

cd ./custom_verl
python3 -m recipe.curio_sft.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=60 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0001 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((($MAX_RESPONSE_LENGTH + 1024)*1)) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.00 \
    +actor_rollout_ref.rollout.val_kwargs.cal_val_entropy=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=178 \
    trainer.test_freq=59 \
    trainer.total_epochs=3 \
    +trainer.rollout_data_dir="${CKPTS_DIR}/rollout_data" \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.actor.policy_loss.loss_mode=custom_loss \
    actor_rollout_ref.actor.policy_loss.loss_remove_token_mean=False \
    actor_rollout_ref.actor.policy_loss.loss_remove_clip=False \
    +actor_rollout_ref.actor.policy_loss.algo_type="curio_sft" \
    +actor_rollout_ref.actor.use_self_distillation=True \
    +actor_rollout_ref.actor.use_adaptive_tau=True \
    +actor_rollout_ref.model.ref_path=$MODEL_PATH \
    +actor_rollout_ref.model.exploration_strategy="fixed_external" \
    +actor_rollout_ref.actor.sync_ref_model=True \
    +actor_rollout_ref.actor.ref_model_sync_steps=5 \
    +actor_rollout_ref.actor.ref_model_mixup_alpha=0.99 \
    +reward_model.reward_impl_version=1 
