<div align="center">

# Post-Trained MoE Can Skip Half Experts via Self-Distillation

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.18643)  [![Github](https://img.shields.io/badge/ZEDA-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TsinghuaC3I/ZEDA) [![HuggingFace](https://img.shields.io/badge/HuggingFace-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/collections/TsinghuaC3I/zeda)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#zeda" style="text-decoration: none; font-weight: bold;">✨ ZEDA</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a> •
    <a href="#acknowledgements" style="text-decoration: none; font-weight: bold;">💖 Acknowledgements</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

> Fully trained Mixture-of-Experts (MoE) models are expensive to serve. Dynamic variant of MoE reduces computation by adjusting the activated experts in an input-dependent manner, while most existing dynamic MoE methods rely on pre-training from scratch or task-specific adaptation.
> 
> **In this paper, we introduce ZEDA, a low-cost framework that transforms post-trained static MoE models into efficient dynamic ones, eliminating over 50% of expert FLOPs at marginal accuracy loss.**

# 🎉News

- **[2026-05-19]** We introduce **Zero-Expert Self-Distillation Adaptation (ZEDA)**.

# 📖Introduction

We introduce **Zero-Expert Self-Distillation Adaptation (ZEDA)**, a low-cost framework that transforms post-trained static MoE models into efficient dynamic ones without substantially sacrificing their established capabilities. ZEDA targets the practical deployment scenario where MoE models have already undergone expensive pre-training and post-training, and further inference-cost reduction is desired after the main training pipeline is finalized. 

To stabilize this architectural conversion, ZEDA injects parameter-free zero-output experts into each MoE layer and adapts the augmented model through **two-stage self-distillation**, utilizing the original MoE as a frozen teacher and applying a **group-level balancing loss**. 
On Qwen3-30B-A3B and GLM-4.7-Flash across 11 benchmarks spanning math, code, and instruction following, ZEDA eliminates over 50% of expert FLOPs at marginal accuracy loss. It outperforms the strongest dynamic MoE baseline by 6.1 and 4.0 points on the two models, and delivers ~1.20× end-to-end inference speedup.

<p align="center">
   <img src="figs/zeda.png" alt="Overview of Unified Post-Training Framework." style="width: 100%;">
</p>



# ✨ZEDA

> ZEDA first injects zero experts into a post-trained MoE, architecturally converting it into a dynamic one, and then adapts it through two-stage self-distillation with the original MoE as a fixed teacher.

ZEDA introduces parameterless zero experts, whose outputs are identically zero, into the existing expert pool of a post-trained MoE model. This expands the router candidate pool with zero-computation experts while the activation number remains unchanged, naturally reducing active normal experts. The augmented model is then adapted through a two-stage self-distillation process:
   - **SFT Stage**: Trains the student on responses sampled from the teacher (original MoE).
   - **OPD Stage**: Shifts to on-policy learning, where responses are sampled from the current student and the teacher supplies token-level targets via reverse KL.

ZEDA incorporates the Group Auxiliary Loss $\mathcal{L}_{GA}$ to regulate the relative activation frequency between normal experts and zero experts, while preserving the learned routing structures among normal experts. The loss is defined as:

```math
\mathcal{L}_{GA} = \alpha \cdot \frac{N + N_Z \cdot w}{K} \cdot \left( \frac{f_{\mathcal{E}} \cdot P_{\mathcal{E}}}{N} + \frac{f_{\mathcal{Z}} \cdot P_{\mathcal{Z}}}{N_Z \cdot w} \right)
```


# 🚀Getting Started

To run ZEDA, follow these steps:

### Env Setup

ZEDA is built upon large-scale MoE training and serving codebases, including [slime](https://github.com/THUDM/slime), [SGLang](https://github.com/sgl-project/sglang), and [Megatron](https://github.com/NVIDIA/Megatron-LM). Please use the Docker image [`slimerl/slime:20251113-v1`](https://hub.docker.com/r/slimerl/slime) released by [slime](https://github.com/THUDM/slime):
```bash
# Pull the image
docker pull slimerl/slime:20251113-v1

# Start the container
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```


After pull and start the docker container, you simply need to install our modified versions of SGLang and slime:
```bash
cd zeda/sglang/python
pip install -e . --no-deps
git apply ../../slime/docker/patch/latest/sglang.patch

cd ../../transformers
pip install -e . --no-deps

cd ../slime
pip install -e . --no-deps
```

### Data Preparation

ZEDA uses 60k prompts including math, code, and chat data, and the corresponding self-distillation rollouts. 
  - **Prompts**: The prompts are used for rollout and OPD. The prompts are chosen from [AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) and [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset), and we release them in [ZEDA-prompts-60k](https://huggingface.co/datasets/TsinghuaC3I/ZEDA/blob/main/ZEDA-prompts-60k.jsonl).
  - **Rollouts**: The rollouts are used for SFT. You need to use the specific post-trained MoE model intended for adaptation to perform the rollout. You can also directly utilize our released rollout results [ZEDA-Qwen3-30B-A3B-rollout-60k](https://huggingface.co/datasets/TsinghuaC3I/ZEDA/blob/main/ZEDA-Qwen3-30B-A3B-rollout-60k.jsonl) and [ZEDA-GLM-4.7-Flash-rollout-60k](https://huggingface.co/datasets/TsinghuaC3I/ZEDA/blob/main/ZEDA-GLM-4.7-Flash-rollout-60k.jsonl).

After downloading the data, please put them in the `data` folder.


### Model Preparation

After downloading the specific post-trained MoE model intended for adaptation from Huggingface, please first convert the model into a dynamic one through zero expert injection:
```bash
python scripts/convert-hf-to-ZCE.py --input-dir path_Qwen3-30B-A3B --output-dir path_Qwen3-30B-A3B-dynamic --new-num-experts 192 # for Qwen3-30B-A3B

python scripts/convert-hf-to-ZCE.py --input-dir path_GLM-4.7-Flash --output-dir path_GLM-4.7-Flash-dynamic --new-num-experts 96 # for GLM-4.7-Flash
```

Then modify the `config.json` in the output Dynamic MoE dir, modify the `"architectures"` and add `"use_zce_mask"`, `"zce_nums"`, and `"zce_types"`. For Qwen3-30B-A3B:
```json
"architectures": ["Qwen3MoePlusPlusForCausalLM"],
"use_zce_mask": false,
"zce_nums": [64],
"zce_types": ["zero"]
```

For GLM-4.7-Flash:
```json
"architectures": ["Glm4MoeLitePlusPlusForCausalLM"],
"use_zce_mask": false,
"zce_nums": [32],
"zce_types": ["zero"]
```

Finally, convert the dynamic MoE into the format compatible with Megatron:
```bash
bash scripts/convert_hf_to_torch_dist.sh
```



### Training
ZEDA consists of zero-expert injection, SFT, and OPD. You can run the following scripts to start the adaptation pipeline:

```bash
# For Qwen3-30B-A3B
bash scripts/train_zeda_qwen_sft.sh # SFT
bash scripts/convert_torch_dist_to_hf.sh # Convert Model
bash scripts/run_teacher_server.sh # Start Teacher Server
bash scripts/train_zeda_qwen_opd.sh # After the teacher server starts, run OPD

# For GLM-4.7-Flash
bash scripts/train_zeda_glm_sft.sh # SFT
bash scripts/convert_torch_dist_to_hf.sh # Convert Model
bash scripts/run_teacher_server.sh # Start Teacher Server
bash scripts/train_zeda_glm_opd.sh # After the teacher server starts, run OPD
```

### Evaluation

We provide a unified evaluation pipeline covering the released math, code, instruction-following, and science benchmarks. To reproduce the reported results, first download the benchmark data from Hugging Face to `evaluation/benchmark`:

```bash
cd ZEDA
hf download TsinghuaC3I/ZEDA-Evaluation \
  --repo-type dataset \
  --local-dir evaluation/benchmark
```

Then install the required evaluation dependencies:

```bash
pip install -r requirements.txt
```

Next, configure `evaluation/run_sglang_server.sh` by specifying:

- `MODEL_PATH`: path to the evaluated model
- `TP_SIZE`: tensor parallel size
- `PORT`: server port

Launch the SGLang server with:

```bash
bash evaluation/run_sglang_server.sh
```

Once the server is ready, configure `evaluation/run_evaluation.sh` by setting:

- `MODEL_NAME`: model name used in output filenames
- `MODEL_PATH`: path to the evaluated model
- `SERVER_URL`: server endpoint, for example `http://0.0.0.0:$PORT/generate`

Then start the evaluation pipeline:

```bash
bash evaluation/run_evaluation.sh
```

The script iterates over all released benchmarks, stores raw generations in `evaluation/raw_output/`, and finally invokes `compute_reward.py` to aggregate the evaluation metrics.

### Models and Datasets

We release our adapted dynamic MoE models and rollout data in Huggingface:

| **Model**                          | **Huggingface** |  **Base Model** |
|-----------------------------------|------------------|------------------|
| ZEDA-Qwen3-30B-A3B-Dynamic | https://huggingface.co/TsinghuaC3I/ZEDA-Qwen3-30B-A3B-Dynamic |  Qwen3-30B-A3B |
| ZEDA-GLM-4.7-Flash-Dynamic | https://huggingface.co/TsinghuaC3I/ZEDA-GLM-4.7-Flash-Dynamic | GLM-4.7-Flash |


| **Rollout Data**                          | **Huggingface** |
|-----------------------------------|------------------|
| ZEDA-Qwen3-30B-A3B-rollout-60k | https://huggingface.co/datasets/TsinghuaC3I/ZEDA |
| ZEDA-GLM-4.7-Flash-rollout-60k | https://huggingface.co/datasets/TsinghuaC3I/ZEDA |


# 📊Main Results

ZEDA demonstrates consistent improvements across multiple models and benchmarks:

<p align="center">
  <img src="figs/performance.png" width="90%">
</p>
<p align="center">
  <img src="figs/training_time.png" width="90%">
</p>
<p align="center">
  <img src="figs/inference_time.png" width="90%">
</p>

# 💖Acknowledgements
Our project mainly builds upon [slime](https://github.com/THUDM/slime), [SGLang](https://github.com/sgl-project/sglang), and [Megatron](https://github.com/NVIDIA/Megatron-LM). We leverage the datasets of [AceReason](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) and [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset), and backbone models of [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) and [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash). We are grateful for these significant open-source contributions.

# 📨Contact

For questions about this work, please contact:

- Xingtai Lv: lvxt24@mails.tsinghua.edu.cn


# 🎈Citation

If you find this work helpful, please cite our paper:

```bibtex
@misc{lv2026posttrainedmoeskiphalf,
      title={Post-Trained MoE Can Skip Half Experts via Self-Distillation}, 
      author={Xingtai Lv and Li Sheng and Kaiyan Zhang and Yichen You and Siyan Gao and Xueheng Luo and Yuxin Zuo and Yuchen Fan and Junlin Yang and Ganqu Cui and Bingning Wang and Fan Yang and Youbang Sun and Ning Ding and Bowen Zhou},
      year={2026},
      eprint={2605.18643},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.18643}, 
}
```
