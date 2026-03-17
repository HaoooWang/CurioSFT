<div align="center">

# CurioSFT

### Learning While Staying Curious: Entropy-Preserving Supervised Fine-Tuning via Adaptive Self-Distillation for Large Reasoning Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.02244-b31b1b.svg)](https://arxiv.org/abs/2602.02244)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/Hao0oWang/curiosft)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-training-pipeline">Training</a> |
  <a href="#-evaluation">Evaluation</a> |
  <a href="#-citation">Citation</a>
</p>

</div>

---

## Introduction

The standard post-training recipe for large reasoning models — Supervised Fine-Tuning followed by Reinforcement Learning (SFT-then-RL) — often limits the benefits of the RL stage. While SFT imitates expert demonstrations, it inevitably drives the model toward overconfidence and reduces generation diversity ("entropy collapse"), leaving RL with a narrowed solution space to explore.

We propose **CurioSFT**, an entropy-preserving SFT method designed to enhance exploration capabilities through intrinsic curiosity. It consists of two key components:

- **Self-Exploratory Distillation**: Distills the model toward a self-generated, temperature-scaled teacher to encourage exploration within its valid capability.
- **Entropy-Guided Temperature Selection**: Adaptively adjusts distillation strength based on token-level uncertainty to amplify exploration at reasoning tokens while stabilizing factual tokens.

Extensive experiments on mathematical reasoning tasks demonstrate that **CurioSFT outperforms vanilla SFT by 2.5 points (ID) and 2.9 points (OOD)**. Crucially, the preserved exploration capability translates into significant gains in the subsequent RL stage, yielding an **average improvement of 5.0 points**.

## Resources

| Resource | Link |
|:---|:---|
| Paper | [![arXiv](https://img.shields.io/badge/arXiv-2602.02244-b31b1b.svg)](https://arxiv.org/abs/2602.02244) |
| CurioSFT Model (SFT stage) | [Hao0oWang/CurioSFT-Qwen2.5-Math-7B-SFT](https://huggingface.co/Hao0oWang/CurioSFT-Qwen2.5-Math-7B-SFT) |
| CurioSFT Model (RL stage) | [Hao0oWang/CurioSFT-Qwen2.5-Math-7B-RL](https://huggingface.co/Hao0oWang/CurioSFT-Qwen2.5-Math-7B-RL) |
| Training Data | [Hao0oWang/CurioSFT_Data](https://huggingface.co/datasets/Hao0oWang/CurioSFT_Data) |

---

## Installation

### Prerequisites
- **Environment**: Python >= 3.10
- **Hardware**: Scripts are configured for **8x H800** by default. Please adjust batch sizes and parallelism settings according to your resources.

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/HaoooWang/CurioSFT.git
   cd CurioSFT
   ```
2. **Create Conda Environment**
   ```bash
   conda create -n curiosft python=3.10
   conda activate curiosft
   ```

3. **Install dependencies**

   We use a custom version of `verl`. Install it along with other requirements:
   ```bash
   cd custom_verl
   pip install -e .
   bash scripts/install_vllm_sglang_mcore.sh
   pip install -r requirements.txt
   ```
   > All modifications are gathered in `custom_verl/recipe/curio_sft`. Please refer to this directory for implementation details.

4. **Prepare data and models**

   Download the dataset and base model from Hugging Face:
   ```bash
   cd CurioSFT
   bash scripts/prepare_data_and_models.sh
   ```
   > **Note**: This script downloads data to `data/` and the base model to `models/`. Ensure you have `huggingface-cli` installed (`pip install -U huggingface_hub`) and are logged in (`huggingface-cli login`) for gated models.

---

## Training Pipeline

The training process consists of two stages: **SFT (CurioSFT)** and **GRPO (RL)**.

### Stage 1: SFT (CurioSFT)

1. **Start the Reward Server** (Required)

   ```bash
   cd CurioSFT
   tmux new-session -d -s reward_server_$(date +%Y%m%d_%H%M%S) 'python -m custom_verl.recipe.curio_sft.reward_function.reward_api'
   ```

2. **Run SFT**
   ```bash
   cd CurioSFT
   TRAIN_CKPT_PATH=/path/to/your/model bash scripts/sft.sh
   ```
   - **Base Model**: [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)
   - **Output**: Checkpoints are saved to `CurioSFT/exp_results/`.

### Stage 2: Reinforcement Learning (GRPO)

1. **Ensure Reward Server is running** (see above).

2. **Run RL**

   ```bash
   cd CurioSFT
   TRAIN_CKPT_PATH=/path/to/your/model bash scripts/rl.sh
   ```

---

## Evaluation

We provide comprehensive evaluation scripts to reproduce the benchmark results reported in the paper, covering both **In-Distribution** (Mathematical Reasoning) and **Out-of-Distribution** (General Reasoning) tasks.

1. **Download Trained Models** (Optional)

   You can download the trained checkpoints for both stages from Hugging Face:
   ```bash
   # Download the CurioSFT model
   huggingface-cli download Hao0oWang/CurioSFT-Qwen2.5-Math-7B-SFT --local-dir models/CurioSFT-7B-SFT

   # Download the CurioSFT-then-RL model
   huggingface-cli download Hao0oWang/CurioSFT-Qwen2.5-Math-7B-RL --local-dir models/CurioSFT-7B-RL
   ```

2. **Start Reward Server**
   ```bash
   tmux new-session -d -s reward_server_$(date +%Y%m%d_%H%M%S) 'python -m custom_verl.recipe.curio_sft.reward_function.reward_api'
   ```

3. **Run Evaluation**
   - **Mathematical Reasoning** (In-Distribution):
     ```bash
     EVAL_CKPT_PATH=/path/to/your_ckpt_path bash scripts/3_eval_math.sh
     ```
   - **GPQA-Diamond, ARC-Challenge** (Out-of-Distribution):
     ```bash
     EVAL_CKPT_PATH=/path/to/your_ckpt_path bash scripts/4_eval_ood_gpqa_arc.sh
     ```
   - **MMLU-Pro** (Out-of-Distribution):
     ```bash
     EVAL_CKPT_PATH=/path/to/your_ckpt_path bash scripts/4_eval_ood_mmlu.sh
     ```

---

## Citation

If you find **CurioSFT** useful in your research, please kindly cite our paper:

```bibtex
@misc{curioSFT,
      title={Learning While Staying Curious: Entropy-Preserving Supervised Fine-Tuning via Adaptive Self-Distillation for Large Reasoning Models},
      author={Hao Wang and Hao Gu and Hongming Piao and Kaixiong Gong and Yuxiao Ye and Xiangyu Yue and Sirui Han and Yike Guo and Dapeng Wu},
      year={2026},
      eprint={2602.02244},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02244},
}
```

## Contact

For questions or feedback, please feel free to [open an issue](https://github.com/HaoooWang/CurioSFT/issues) or contact [hao.wang@my.cityu.edu.hk](mailto:hao.wang@my.cityu.edu.hk).
