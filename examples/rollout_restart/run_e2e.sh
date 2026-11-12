#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
#
# Rollout-restart e2e demo (design §11.2): 4-NPU Megatron training + 4-NPU vLLM
# standalone rollout on one Ascend 910B node, EngineCore SIGKILL mid-rollout,
# recovery verification, and an enabled=false control group that is expected
# to crash (the current behavior, demonstrated live).
#
# Usage:
#   bash examples/rollout_restart/run_e2e.sh                       # full demo (recover + control)
#   PHASE=recover  bash examples/rollout_restart/run_e2e.sh        # recovery leg only
#   PHASE=control  bash examples/rollout_restart/run_e2e.sh        # control leg only
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE=${PHASE:-all}                       # all | recover | control

MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}
WORKDIR=${WORKDIR:-$(mktemp -d /tmp/rollout_restart_demo.XXXXXX)}
EXPERIMENT=${EXPERIMENT:-rollout_restart_demo_$(date +%Y%m%d_%H%M%S)}

NUM_GPUS=${NUM_GPUS:-8}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-4}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=${TRAIN_PP:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}
SAVE_FREQ=${SAVE_FREQ:-2}                 # small so a route-A checkpoint exists early
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}

# HCCL timeouts: bound the stuck-collective teardown path (design §6.5.1).
# The manager's bounded_abort_s (120s) drives the teardown; HCCL_EXEC_TIMEOUT
# is the safety net beyond it.
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_EXEC_TIMEOUT=600
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_USE_V1=1

n_gpus_training=$((NUM_GPUS - N_GPUS_ROLLOUT))
if (( n_gpus_training != N_GPUS_TRAINING )); then
    echo "NUM_GPUS (${NUM_GPUS}) - N_GPUS_ROLLOUT (${N_GPUS_ROLLOUT}) must equal N_GPUS_TRAINING (${N_GPUS_TRAINING})"
    exit 1
fi

run_leg() {
    local leg=$1            # recover | control
    local restart_enabled=$2
    local log_file="${WORKDIR}/driver_${leg}.log"
    local record_file="${WORKDIR}/injection_${leg}.json"
    local expect=recover
    if [[ "${leg}" == "control" ]]; then expect=crash; fi

    local checkpoint_root="${WORKDIR}/checkpoints/${leg}"

    local params=(
        trainer.use_v1=True
        trainer.v1.trainer_mode=separate_async
        trainer.v1.separate_async.num_warmup_batches=1
        trainer.v1.separate_async.parameter_sync_step=4
        trainer.v1.separate_async.hybrid_rollout.enable_switch=False
        transfer_queue.enable=True
        data.train_files="${TRAIN_FILE}"
        data.val_files="${VAL_FILE}"
        data.prompt_key=prompt
        data.truncation=left
        data.filter_overlong_prompts=False
        data.max_prompt_length=512
        data.max_response_length=1024
        data.train_batch_size=${TRAIN_BATCH_SIZE}
        data.gen_batch_size=${GEN_BATCH_SIZE}
        algorithm.adv_estimator=grpo
        algorithm.use_kl_in_reward=False
        actor_rollout_ref.model.path="${MODEL_PATH}"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.optim.lr_warmup_steps=-1
        actor_rollout_ref.actor.ppo_mini_batch_size=4
        actor_rollout_ref.actor.ppo_epochs=1
        actor_rollout_ref.actor.use_dynamic_bsz=False
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.actor.entropy_coeff=0
        actor_rollout_ref.actor.strategy=megatron
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
        ++actor_rollout_ref.actor.megatron.override_transformer_config.use_triton_gdn=True
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
        actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
        actor_rollout_ref.actor.megatron.param_offload=True
        actor_rollout_ref.actor.megatron.optimizer_offload=True
        actor_rollout_ref.actor.use_torch_compile=False
        actor_rollout_ref.ref.strategy=megatron
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
        ++actor_rollout_ref.ref.megatron.override_transformer_config.use_triton_gdn=True
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
        actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.mode=async
        actor_rollout_ref.rollout.nnodes=1
        actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6
        actor_rollout_ref.rollout.n=1
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
        actor_rollout_ref.rollout.restart.enabled=${restart_enabled}
        actor_rollout_ref.rollout.restart.checkpoint_dir="${checkpoint_root}"
        reward.reward_manager.name=dapo
        trainer.logger='["console"]'
        trainer.project_name=rollout-restart-demo
        trainer.experiment_name="${EXPERIMENT}_${leg}"
        trainer.val_before_train=False
        trainer.test_freq=-1
        trainer.save_freq=${SAVE_FREQ}
        trainer.default_local_dir="${checkpoint_root}"
        trainer.resume_mode=disable
        trainer.nnodes=1
        trainer.n_gpus_per_node=${n_gpus_training}
        trainer.total_epochs=1
        trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    )

    echo "=== [${leg}] restart.enabled=${restart_enabled}: launching trainer (log: ${log_file}) ==="
    python3 -m verl.trainer.main_ppo --config-path=config --config-name=ppo_megatron_trainer.yaml \
        "${params[@]}" >"${log_file}" 2>&1 &
    local driver_pid=$!

    # Inject an EngineCore SIGKILL once a committed checkpoint exists (route-A
    # weight source) and the rollout has been live for a settle window.
    python3 "${SCRIPT_DIR}/inject_engine_kill.py" \
        --wait-checkpoint "${checkpoint_root}" \
        --settle-s "${SETTLE_S:-20}" \
        --out "${record_file}" || true

    # Control leg: the job is expected to die shortly after the kill.
    if [[ "${leg}" == "control" ]]; then
        local wait_deadline=$((SECONDS + 900))
        while kill -0 "${driver_pid}" 2>/dev/null; do
            if (( SECONDS > wait_deadline )); then
                echo "=== [${leg}] driver still alive 15min after injection (unexpected for the control group) ==="
                break
            fi
            sleep 10
        done
    fi

    wait "${driver_pid}" || echo "=== [${leg}] driver exited nonzero (expected for the control group) ==="

    python3 "${SCRIPT_DIR}/verify_recovery.py" \
        --log "${log_file}" \
        --record "${record_file}" \
        --expect "${expect}" \
        --final-step ${TOTAL_TRAINING_STEPS}
    echo "=== [${leg}] verification passed ==="
}

if [[ "${PHASE}" == "all" || "${PHASE}" == "recover" ]]; then
    run_leg recover True
fi
if [[ "${PHASE}" == "all" || "${PHASE}" == "control" ]]; then
    run_leg control False
fi

echo "=== rollout-restart demo finished; artifacts under ${WORKDIR} ==="
