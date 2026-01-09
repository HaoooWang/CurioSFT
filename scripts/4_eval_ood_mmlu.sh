set -x
ulimit -n 65535

export HYDRA_FULL_ERROR=1
export NO_PROXY="127.0.0.1,localhost"
export VLLM_USE_V1=1

extract_model_info() {
    local full_path="$1"
    
    if echo "$full_path" | grep -q "global_step"; then
        local step_num=$(echo "$full_path" | grep -o 'global_step_[0-9]*' | grep -o '[0-9]*')
        local global_step_parent=$(echo "$full_path" | sed 's|/global_step_[0-9]*.*||' | xargs basename)
        local exp_name="$global_step_parent"
        local new_model_name="${exp_name}_step${step_num}"

    elif echo "$full_path" | grep -q "checkpoint-"; then
        local step_num=$(echo "$full_path" | grep -o 'checkpoint-[0-9]*' | grep -o '[0-9]*')
        local ckpt_parent=$(echo "$full_path" | sed 's|/checkpoint-[0-9]*.*||' | xargs basename)
        local exp_name="$ckpt_parent"
        local new_model_name="${exp_name}_step${step_num}"
    else
        local new_model_name=$(basename "$full_path")
    fi
    
    echo "$new_model_name"
}

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_CKPT_PATH="${EVAL_CKPT_PATH:-"${PROJECT_DIR}/models/CurioSFT-7B-SFT"}"

CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"
TRAIN_FILE="$PROJECT_DIR/data/train.parquet"
VAL_FILE="$PROJECT_DIR/data/test.mmlu_pro.parquet"

model_name=$(extract_model_info $EVAL_CKPT_PATH | tail -n 1)
MODEL_PATH="${EVAL_CKPT_PATH}"
MAX_RESPONSE_LENGTH=8192
WANDB_PROJECT="CurioSFT"
EXP_NAME="eval_$model_name"
CKPTS_DIR="$PROJECT_DIR/eval_results/mmlu_pro/${EXP_NAME}"

mkdir -p ${CKPTS_DIR}
ray start --head

cd ./custom_verl
python3 -m recipe.curio_sft.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
    trainer.logger='["console"]' \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=12 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    +trainer.rollout_data_dir="${CKPTS_DIR}/rollout_data" \
    +reward_model.reward_impl_version=0 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.actor.policy_loss.loss_mode=custom_loss \
    +actor_rollout_ref.actor.policy_loss.algo_type="CurioSFT" \
    +actor_rollout_ref.model.exploration_strategy="naive" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \


