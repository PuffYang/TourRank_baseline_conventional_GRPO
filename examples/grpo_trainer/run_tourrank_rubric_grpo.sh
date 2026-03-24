#!/usr/bin/env bash
set -euo pipefail
set -x

TRAIN_FILE=${TRAIN_FILE:-"/Users/yangzixuan/Desktop/TourRank/DR-Tulu/data/RL/verl_rubric_grpo/train.parquet"}
VAL_FILE=${VAL_FILE:-"/Users/yangzixuan/Desktop/TourRank/DR-Tulu/data/RL/verl_rubric_grpo/val.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
N_ROLLOUTS=${N_ROLLOUTS:-4}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    reward.reward_manager.name=rubric_judge \
    +reward.rubric_judge.n_rollouts="${N_ROLLOUTS}" \
    +reward.rubric_judge.model=gpt-4o \
    +reward.rubric_judge.temperature=0.0 \
    +reward.rubric_judge.max_tokens=1200 \
    +reward.rubric_judge.timeout=200 \
    +reward.rubric_judge.api_key="b1b6cfd6240c446dbbe8ca087ca7fc02" \
    +reward.rubric_judge.api_version="2024-06-01" \
    +reward.rubric_judge.azure_endpoint="https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-06-01" \
    trainer.logger='["console"]' \
    trainer.project_name='tourrank_grpo' \
    trainer.experiment_name='tourrank_rubric_gpt4o_judge' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    "$@"
