# For Qwen3-30B-A3B
source scripts/models/qwen3-30B-A3B-ZCE.sh
PYTHONPATH=zeda/Megatron-LM-dynn/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint hf-model-path \
   --save megatron-model-path

# For GLM-4.7-Flash
source scripts/models/glm4.7-30B-A3B-ZCE.sh
PYTHONPATH=zeda/Megatron-LM-dynn/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint hf-model-path \
   --save megatron-model-path