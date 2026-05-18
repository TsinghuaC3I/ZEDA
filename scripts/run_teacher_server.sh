# For Qwen3-30B-A3B
python -m sglang.launch_server --model-path path_Qwen3-30B-A3B --tp 2 --ep 2 --dp 4 --port 25431 --host 0.0.0.0 --attention-backend fa3

# For GLM-4.7-Flash
python -m sglang.launch_server --model-path path_GLM-4.7-Flash --tp 2 --ep 2 --dp 4 --port 25431 --host 0.0.0.0 --attention-backend fa3
