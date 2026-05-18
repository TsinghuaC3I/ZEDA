import os
import torch
import json
import argparse
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import shutil


def gaussian_init(t: torch.Tensor, old_t: torch.Tensor) -> torch.Tensor:
    '''
    Gaussian initialization for the gate matrix. Mean & std are calculated from the old weights.
    old_t: [old_num_experts, hidden]
    t: [new_num_experts, hidden]
    mean: [hidden]
    std: [hidden]
    '''
    # consider one dimension case
    if old_t.ndim == 2:
        assert old_t.shape[1] == t.shape[1], "The hidden dimension of the old and new weights must be the same"

    mean = old_t.mean(dim=0)
    std = old_t.std(dim=0)
    
    w = torch.randn(t.shape, dtype=old_t.dtype, device=old_t.device) * std + mean

    w[:old_t.shape[0]] = old_t
    return w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/mnt/shared-storage-user/p1-shared/shengli/llms/hf/GLM-4.7-Flash")
    parser.add_argument("--output-dir", type=str, default="/mnt/shared-storage-user/p1-shared/shengli/llms/hf/GLM-4.7-Flash-ZCE")
    parser.add_argument("--new-num-experts", type=int, default=96, help="Target number of experts for the router output dimension")
    parser.add_argument("--router-key-patterns", type=str, nargs="+", default=["mlp.gate.weight", "mlp.gate.e_score_correction_bias"], help="Substring to identify router weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target router experts: {args.new_num_experts}")

    # 1. Copy and verify config
    config_path = os.path.join(args.input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        old_num_experts = config.get("num_experts", None) or config.get("n_routed_experts", None)
        print(f"Original num_experts in config: {old_num_experts}")
        
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print("Config copied to output directory (unchanged).")
    else:
        raise ValueError(f"old_num_experts not found in config.json")

    # 2. Process weights
    # Check for safetensors
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".safetensors")]
    if not files:
        print("No .safetensors files found in input directory.")
        return

    # Copy other files (tokenizer, etc.)
    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".safetensors") and filename != "config.json":
            src = os.path.join(args.input_dir, filename)
            dst = os.path.join(args.output_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    for file in tqdm(files, desc="Processing weights"):
        file_path = os.path.join(args.input_dir, file)
        state_dict = load_file(file_path)
        new_state_dict = {}
        modified_count = 0
        
        for key, tensor in state_dict.items():
            matched = False
            for router_key_pattern in args.router_key_patterns:
                if router_key_pattern in key:
                    matched = True
                    assert tensor.shape[0] == old_num_experts, f"The number of experts in {key} is not equal to the original number of experts"
                    if tensor.shape[0] < args.new_num_experts:
                        if "bias" in key:
                            print(f"Expanding bias {key}: shape {tensor.shape} -> ({args.new_num_experts},)")
                            new_shape = (args.new_num_experts,)
                        else:
                            print(f"Expanding {key}: {tensor.shape} -> ({args.new_num_experts}, {tensor.shape[1]})")
                            new_shape = (args.new_num_experts, tensor.shape[1])

                        new_tensor = torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)
                        new_tensor = gaussian_init(new_tensor, tensor)
                        assert torch.equal(new_tensor[:old_num_experts], tensor)
                        new_state_dict[key] = new_tensor
                        modified_count += 1
                    else:
                        print(f"Skipping {key}: shape {tensor.shape} already >= target {args.new_num_experts}")
                        new_state_dict[key] = tensor
                    break
            if not matched:
                new_state_dict[key] = tensor
        
        print(f"Saving {file} with {modified_count} modified tensors...")
        save_file(new_state_dict, os.path.join(args.output_dir, file))

    print("Done.")

if __name__ == "__main__":
    main()
