#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Wandb
export WANDB_MODE=offline

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

######### Parameters ###########
hf_checkpoint_path=hf-model-path
megatron_checkpoint_path=megatron-model-path
###############################

# Training args
aux_loss_coeff=0.1
moe_zero_expert_weight=2.0
zero_experts_type=zero
experiment_name=Qwen3-30B-A3B-ZCE-sft-${aux_loss_coeff}-weight-${moe_zero_expert_weight}-type-${zero_experts_type}

# Local path
data_path=data/ZEDA-Qwen3-30B-A3B-rollout-60k.jsonl
save_checkpoint_path=results/${experiment_name}
megatron_code_path=zeda/Megatron-LM

source "scripts/models/qwen3-30B-A3B-ZCE.sh"

CKPT_ARGS=(
   --hf-checkpoint ${hf_checkpoint_path}
   --ref-load ${megatron_checkpoint_path}
   # --load
   --save ${save_checkpoint_path}
   --save-interval 500
   --no-save-optim
   --no-save-rng
)

MOE_ARGS=(
   --zero-experts-type ${zero_experts_type}
   --use-moe-router-tracker
   --moe-router-load-balancing-type grouped_aux_loss
   --moe-zero-expert-weight ${moe_zero_expert_weight}
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${data_path}
   --input-key messages
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)


PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 65536
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style cosine
   --min-lr 0
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --moe-aux-loss-coeff ${aux_loss_coeff}

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project dynn-experiments
   --wandb-group ${experiment_name}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${megatron_code_path}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 zeda/slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${MOE_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}