# MLX LoRA Fine-tuning Engine

Engine for fine-tuning LLM models using LoRA and MLX on Apple Silicon.

## Features

- 🚀 **Fine-tuning with LoRA**: Efficient training using Low-Rank Adaptation.
- 🍎 **MLX Optimized**: Leverages Apple Silicon hardware to the fullest.
- 🤗 **Hugging Face Integration**: Download and upload models directly from/to the HF Hub.
- 💾 **Checkpoint Management**: Save and restore checkpoints during training.
- 📊 **Data Utils**: Tools to prepare suitable training and validation datasets.

## Installation

```bash
# Clone the repository
git clone https://github.com/santos-sanz/mlx-lora-finetune-template.git
cd mlx-lora-finetune-template

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN
```

## Configuration

Copy `.env.example` to `.env` and configure the variables:

```bash
HF_TOKEN=your_hugging_face_token
HF_REPO_ID=your-username/model-name
MODEL_NAME=meta-llama/Llama-3.2-1B
```

## Usage

### 1. Prepare Data

```bash
python scripts/prepare_data.py \
    --input data/raw/dataset.json \
    --output data/processed \
    --val-split 0.1
```

### 2. Train Model

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --model meta-llama/Llama-3.2-1B \
    --data data/processed
```

### 3. Upload to Hugging Face

```bash
# Upload final model
python scripts/upload_to_hf.py --model outputs/adapters/final

# Upload specific checkpoint
python scripts/upload_to_hf.py --checkpoint outputs/checkpoints/step-1000
```

## Project Structure

```
mlx-lora-finetune-template/
├── src/                    # Main source code
│   ├── config.py          # Configurations (LoRA, training, model)
│   ├── data_utils.py      # Data processing utilities
│   ├── hf_utils.py        # Hugging Face integration
│   ├── model_utils.py     # Model loading and management
│   └── trainer.py         # Training engine
├── scripts/               # Executable scripts
│   ├── train.py          # Training script
│   ├── prepare_data.py   # Data preparation script
│   └── upload_to_hf.py   # HF Hub upload script
├── configs/              # YAML configuration files
│   └── default.yaml      # Default configuration
├── data/                 # Training data
│   ├── raw/             # Unprocessed data
│   └── processed/       # Prepared data (train.jsonl, valid.jsonl)
└── outputs/             # Training outputs
    ├── adapters/        # Saved LoRA weights
    ├── checkpoints/     # Intermediate checkpoints
    └── logs/            # Training logs
```

## Reference Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Useful Links

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

## License

MIT License