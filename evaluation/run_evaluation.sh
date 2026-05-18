#!/bin/bash

# ==================== MODEL CONFIG ====================
MODEL_NAME=
MODEL_PATH=
SERVER_URL= # eg. http://0.0.0.0:$SERVER_PORT/generate

# ================== BENCHMARK CONFIG ==================
CODE_BENCHMARKS=(
  humanevalplus
  mbppplus
  livecodebench-v5
  livecodebench-v6
)

MATH_BENCHMARKS=(
  gsm8k
  math-500
  aime-2024
  aime-2025
  aime-2026
)

IF_BENCHMARKS=(
  ifeval
  ifbench
)

SCIENCE_BENCHMARKS=(
  gpqa-diamond
  mmlu-redux-2.0
)

BENCHMARK_LIST=(
  "${CODE_BENCHMARKS[@]}"
  "${MATH_BENCHMARKS[@]}"
  "${IF_BENCHMARKS[@]}"
  "${SCIENCE_BENCHMARKS[@]}"
)


# ==================== SAMPLING PARAM ====================
DEFAULT_N=1
DEFAULT_LEN=38912
DEFAULT_CONCURRENCY=64
DEFAULT_TEMPERATURE=0.6
DEFAULT_TOP_K=20
DEFAULT_TOP_P=0.95

declare -A BENCHMARK_N_MAP=(
  [livecodebench-v5]=8
  [livecodebench-v6]=8
  [humanevalplus]=8
  [mbppplus]=8
  [gpqa-diamond]=8
)

declare -A BENCHMARK_LEN_MAP=(
  [gpqa-diamond]=32768
  [mmlu-redux-2.0]=32768
)

for BENCHMARK_NAME in "${BENCHMARK_LIST[@]}"; do
  BENCHMARK_N="${BENCHMARK_N_MAP[$BENCHMARK_NAME]:-$DEFAULT_N}"
  BENCHMARK_LEN="${BENCHMARK_LEN_MAP[$BENCHMARK_NAME]:-$DEFAULT_LEN}"
  echo "========== Running benchmark: ${BENCHMARK_NAME} (n=${BENCHMARK_N}) =========="
  python rollout_manager.py \
    --input  ./benchmark/${BENCHMARK_NAME}.jsonl \
    --output ./raw_output/${BENCHMARK_NAME}-${MODEL_NAME}.jsonl \
    --raw-url $SERVER_URL \
    --api-mode raw \
    --enable-routed-experts-stats \
    --model $MODEL_NAME \
    --model-path $MODEL_PATH \
    --concurrency $DEFAULT_CONCURRENCY \
    --n $BENCHMARK_N \
    --max-tokens $BENCHMARK_LEN \
    --request-timeout 3600.0 \
    --temperature $DEFAULT_TEMPERATURE \
    --top-k $DEFAULT_TOP_K \
    --top-p $DEFAULT_TOP_P
done

python compute_reward.py --pattern "*$MODEL_NAME.jsonl"