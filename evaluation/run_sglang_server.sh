#!/bin/bash
export MODEL_PATH=
export TP_SIZE=
export PORT=

python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp $TP_SIZE \
    --ep $TP_SIZE \
    --port $PORT \
    --host 0.0.0.0 \
    --attention-backend fa3 \
    --enable-return-routed-experts