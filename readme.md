# Training-Free Self-Correction for Multimodal Masked Diffusion Models

This repository contains the inference code for [Training-Free Self-Correction for Multimodal Masked Diffusion Models](https://arxiv.org/abs/2602.02927)
by Yidong Ouyang, Panwen Hu, Zhengyan Wan, Zhe Wang, Liyan Xie, Dmitriy Bespalov, Ying Nian Wu, Hongyuan Zha, Qiang Sun.

## Overview

We propose a training-free self-correction framework for multimodal masked diffusion models, enabling high-quality text-to-image generation and multimodal understanding without any additional fine-tuning.

## Requirements

### Environment Setup

```bash
# Create and activate a conda environment
conda create -n mmdm python=3.10
conda activate mmdm

# Install PyTorch (CUDA 12.1 recommended)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### Model Checkpoints

Download the required checkpoints and place them as follows:

```
.
├── checkpoints/          # Multimodal masked diffusion model checkpoint
│   └── ...
└── vae_ckpt/             # VQ-VAE checkpoint
    └── vqvae/
        └── ...
```

The VQ-VAE checkpoint should be compatible with the `diffusers` `VQModel` format. The main model checkpoint should be loadable via `transformers` `AutoTokenizer` / `from_pretrained`.

## Inference

### Text-to-Image Generation

Generate a single image from a text prompt:

```bash
python inference/inference_t2i.py \
    --checkpoint ./checkpoints \
    --vae_ckpt ./vae_ckpt \
    --prompt "A serene mountain landscape at sunrise" \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --temperature 1.0 \
    --seed 42 \
    --output_dir results_t2i
```

**Key arguments:**
| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to model checkpoint |
| `--vae_ckpt` | `./vae_ckpt` | Path to VQ-VAE checkpoint |
| `--prompt` | — | Text prompt |
| `--height` / `--width` | 1024 | Output image resolution |
| `--timesteps` | 64 | Number of diffusion timesteps |
| `--cfg_scale` | 4.0 | Classifier-free guidance scale |
| `--temperature` | 1.0 | Sampling temperature |
| `--seed` | 0 | Random seed (0 = no fixed seed) |

#### Accelerated Inference with KV-Cache

Enable token-level caching for faster generation:

```bash
python inference/inference_t2i.py \
    --checkpoint ./checkpoints \
    --vae_ckpt ./vae_ckpt \
    --prompt "A serene mountain landscape at sunrise" \
    --use-cache \
    --cache_ratio 0.9 \
    --warmup_ratio 0.3 \
    --refresh_interval 5
```

### Multi-GPU Batch Generation (DDP)

For large-scale evaluation across multiple GPUs:

```bash
torchrun --nproc_per_node=8 inference/inference_t2i_ddp.py \
    --checkpoint ./checkpoints \
    --vae_ckpt ./vae_ckpt \
    --prompt_path prompts.jsonl \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --batch_size 1 \
    --output_dir results_ddp \
    --output_json results_ddp/results.json
```

The `--prompt_path` file can be `.jsonl` (one `{"prompt": "..."}` per line), `.json` (list of strings), or `.txt` (one prompt per line).

### Multimodal Understanding (Visual QA)

```bash
python inference/inference_mmu.py \
    --checkpoint ./checkpoints \
    --vae_ckpt ./vae_ckpt \
    --prompt "What is depicted in this image?" \
    --image_path /path/to/image.png \
    --steps 128 \
    --gen_length 1024 \
    --block_length 256 \
    --temperature 0.0 \
    --output_dir results_mmu
```

## Data Pre-tokenization

To pre-tokenize a dataset before training/evaluation:

```bash
# Edit paths in the script first
bash pre_tokenizer/run_pre_token.sh
```

This splits the work across 32 parallel processes (8 GPUs). Supported `--type` values: `t2i`, `edit`, `mmu_single_image`, `mmu_multi_image`.

After pre-tokenization, merge sub-records:

```bash
python pre_tokenizer/concat_record.py \
    --sub_record_dir /path/to/pre_token_output \
    --save_path /path/to/pre_token_output/all_records.json
```

## Project Structure

```
.
├── config.py                    # Global configuration (tokens, generation defaults)
├── requirements.txt
├── model/
│   ├── configuration_llada.py   # Model configuration
│   ├── modeling_llada.py        # Core masked diffusion LM
│   └── modeling_xllmx_dimoo.py  # Multimodal extension
├── generators/
│   ├── image_generation_generator.py           # T2I MaskGit decoding
│   ├── image_to_image_generator.py             # I2I generation
│   ├── image_to_image_generator_trajectory.py  # I2I with trajectory
│   └── text_understanding_generator.py         # MMU generation
├── inference/
│   ├── inference_t2i.py          # Text-to-image (single GPU)
│   ├── inference_t2i_ddp.py      # Text-to-image (multi-GPU DDP)
│   ├── inference_t2i_trajectory.py
│   ├── inference_i2i.py          # Image-to-image
│   └── inference_mmu.py          # Multimodal understanding
├── utils/
│   ├── generation_utils.py       # Sampling utilities
│   ├── image_utils.py            # VQ-VAE encode/decode helpers
│   └── prompt_utils.py           # Prompt template utilities
├── data/
│   └── item_processor.py         # Data loading utilities
└── pre_tokenizer/
    ├── pre_tokenize.py
    ├── concat_record.py
    └── run_pre_token.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
