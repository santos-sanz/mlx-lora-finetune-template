# 🚀 MLX LoRA Fine-tuning Engine

A powerful, user-friendly engine for fine-tuning LLM models using LoRA and MLX on Apple Silicon. Includes a **Streamlit UI** for easy configuration and training.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🚀 **Fine-tuning with LoRA**: Efficient training using Low-Rank Adaptation
- 🍎 **MLX Optimized**: Leverages Apple Silicon hardware (M1/M2/M3) to the fullest
- 🎨 **Streamlit UI**: Web interface with Simple/Advanced modes
- 🤗 **Hugging Face Integration**: Download and upload models directly from/to the HF Hub
- 💾 **Checkpoint Management**: Save and restore checkpoints during training
- 📊 **Data Preparation**: Convert JSON, raw text, or entire folders into training data
- 🤖 **AI-Assisted Data Prep**: Use small LLMs (Qwen3-0.6B) to generate Q&A pairs

## 🖥️ Screenshots

The Streamlit UI provides:
- **Simple Mode**: Pre-configured presets for beginners (Quick Test, Balanced, High Quality)
- **Advanced Mode**: Full control over all LoRA and training hyperparameters
- **Data Preparation**: Process JSON files, raw text, or entire folders
- **HuggingFace Upload**: Push trained models directly to the Hub

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/santos-sanz/mlx-lora-finetune-template.git
cd mlx-lora-finetune-template

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN
```

## 🚀 Quick Start

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Option 2: Command Line

#### 1. Prepare Data

```bash
python scripts/prepare_data.py \
    --input data/raw/dataset.json \
    --output data/processed \
    --val-split 0.1
```

#### 2. Train Model

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --model meta-llama/Llama-3.2-1B \
    --data data/processed
```

#### 3. Upload to Hugging Face

```bash
python scripts/upload_to_hf.py --model outputs/adapters/final
```

## 🎨 Streamlit UI Guide

### Data Preparation Page

| Tab | Description |
|-----|-------------|
| 📋 Structured JSON | Convert JSON/JSONL datasets with instruction-response pairs |
| 📝 Raw Text/Folder | Process transcripts, books, or entire folders of text files |

**Input Options for Raw Text:**
- **📄 Single File**: Load one text file (.txt, .md)
- **📁 Folder**: Process multiple files at once
- **📋 Paste Text**: Paste content directly

**Processing Options:**
- Remove timestamps, URLs, speaker labels
- Adjustable chunk size and overlap
- Output formats: Completion, Q&A, Knowledge, Raw

**🤖 AI Enhancement (Optional):**
Enable to use Qwen3-0.6B for intelligent Q&A or summary generation.

### Training Page

| Mode | Description |
|------|-------------|
| 🎯 Simple | Choose from presets: Quick Test, Balanced, High Quality, Maximum |
| ⚙️ Advanced | Full control over LoRA rank, learning rate, epochs, etc. |

### HuggingFace Upload Page

Upload your trained model or checkpoints directly to the Hugging Face Hub.

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```bash
HF_TOKEN=your_hugging_face_token
HF_REPO_ID=your-username/model-name
MODEL_NAME=meta-llama/Llama-3.2-1B
```

## 📁 Project Structure

```
mlx-lora-finetune-template/
├── app.py                 # 🎨 Streamlit UI application
├── src/                   # Core source code
│   ├── config.py         # Configuration classes
│   ├── data_utils.py     # Data processing utilities
│   ├── hf_utils.py       # Hugging Face integration
│   ├── model_utils.py    # Model loading utilities
│   └── trainer.py        # Training engine
├── scripts/              # CLI scripts
│   ├── train.py         # Training script
│   ├── prepare_data.py  # Data preparation
│   └── upload_to_hf.py  # HF Hub upload
├── tests/               # Test suite
│   └── test_core.py    # Core tests
├── configs/             # YAML configurations
├── data/               # Training data
│   ├── raw/           # Unprocessed data
│   └── processed/     # Prepared data
└── outputs/           # Training outputs
    ├── adapters/      # LoRA weights
    ├── checkpoints/   # Checkpoints
    └── logs/          # Training logs
```

## 🧪 Running Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Training Presets

| Preset | Epochs | LoRA Rank | Learning Rate | Use Case |
|--------|--------|-----------|---------------|----------|
| 🚀 Quick Test | 1 | 4 | 2e-4 | Verify setup works |
| ⚖️ Balanced | 2 | 8 | 1e-4 | Good quality, reasonable time |
| 🎯 High Quality | 3 | 16 | 5e-5 | Better results, longer training |
| 💎 Maximum | 5 | 32 | 2e-5 | Best quality, longest training |

## 🍎 Recommended Models for Apple Silicon

We recommend starting with **Small Language Models (SLMs)**. They are incredibly efficient, allow for very fast iteration cycles, and can be fine-tuned on standard MacBooks without high memory pressure.

For Apple Silicon, we recommend using **small language models (< 6B parameters)** for efficient training.

👉 **[Browse Trending Text Generation Models (< 6B)](https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:0,max:6B&sort=trending)**

**Tips:**
- Start with models under 1B for fast iteration
- Models like SmolLM2, Qwen, and Llama 3.2 work great
- Check the model's memory requirements before training

### 🧠 Why Tiny Models (SLMs)?
Using models under 1B parameters is highly recommended for:
- **Fast Prototyping**: Complete fine-tuning runs in minutes, not hours.
- **Resource Efficiency**: Low heat generation and minimal memory usage.
- **On-Device Deployment**: Perfect for mobile, IoT, or local-only applications.
- **Research**: Easier to understand model behavior and data influence.

## 📚 Data Formats

### Structured JSON (Instruction-Response)

```json
[
  {"instruction": "What is Python?", "response": "A programming language"},
  {"instruction": "Explain LoRA", "response": "Low-Rank Adaptation..."}
]
```

### Raw Text

Any `.txt` or `.md` file. The system will:
1. Clean the text (remove timestamps, URLs, etc.)
2. Chunk into manageable pieces
3. Generate training examples in your chosen format

## 📖 Reference Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [MLX: Machine Learning on Apple Silicon](https://ml-explore.github.io/mlx/)

## 🔗 Useful Links

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.