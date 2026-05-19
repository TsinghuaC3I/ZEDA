# For Qwen3-30B-A3B
source scripts/models/qwen3-30B-A3B-ZCE.sh
PYTHONPATH=zeda/Megatron-LM/ torchrun --nproc-per-node 1 \
   zeda/slime/tools/convert_torch_dist_to_hf.py \
   --input-dir  megatron-model-path \
   --output-dir hf-model-path \
   --origin-hf-dir origin-hf-model-path

# For GLM-4.7-Flash
source scripts/models/glm4.7-30B-A3B-ZCE.sh
PYTHONPATH=zeda/Megatron-LM/ torchrun --nproc-per-node 1 \
   zeda/slime/tools/convert_torch_dist_to_hf.py \
   --input-dir  megatron-model-path \
   --output-dir hf-model-path \
   --origin-hf-dir origin-hf-model-path