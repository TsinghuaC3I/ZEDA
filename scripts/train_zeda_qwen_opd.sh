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
# run
set -ex

######### Parameters ###########
TEACHER_IP="xx.xx.xx.xx" # Use localhost here, you can change it to your IP
HF_MODEL_PATH=hf-model-path
MEGATRON_MODEL_PATH=megatron-model-path
WANDB_KEY=your_wandb_key
###############################


PROMPT_NUM=16
SAMPLE_NUM=2
LR=5e-6

ZCE_WEIGHT=2

WANDB_NAME="Qwen3-30B-A3B-ZCE-opd-${PROMPT_NUM}-${SAMPLE_NUM}-lr${LR}_zce-weight-${ZCE_WEIGHT}-v3data"
megatron_code_path=zeda/Megatron-LM

TEACHER_PORT=25431

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "scripts/models/qwen3-30B-A3B-ZCE.sh"


CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
   --load results/${WANDB_NAME}/
   --save results/${WANDB_NAME}/
   --save-interval 320

   --zero-experts-type zero
)

MOE_ARGS=(
   --zero-experts-type zero
   --moe-router-load-balancing-type grouped_aux_loss
   --moe-zero-expert-weight $ZCE_WEIGHT
   --moe-aux-loss-coeff 0.1
)

ROLLOUT_ARGS=(
   --prompt-data data/ZEDA-prompts-60k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 321
   --rollout-batch-size $PROMPT_NUM
   --n-samples-per-prompt $SAMPLE_NUM
   --rollout-max-response-len 32768
   --rollout-temperature 1

   --global-batch-size 32
   --balance-data
)

RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
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

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator on_policy_distillation
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr $LR
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --zce-rate-add-position showzce
)

WANDB_ARGS=(
   --use-wandb
   --wandb-mode offline
   --wandb-project slime-dev
   --wandb-group ${WANDB_NAME}
   --wandb-dir zeda/slime/wandb/${WANDB_NAME}
   --wandb-key ${WANDB_KEY}
   --disable-wandb-random-suffix
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-expert-parallel-size 2
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --use-rollout-routing-replay
   --use-slime-router
)


# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${megatron_code_path}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 zeda/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${MOE_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}



####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
