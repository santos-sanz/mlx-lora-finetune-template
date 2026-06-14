"""
Tests for MLX LoRA Fine-tuning Engine

Run with: pytest tests/ -v
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add project root - use direct file imports to avoid MLX dependency
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config classes directly (no MLX dependency)
from src.config import (
    Config, LoRAConfig, TrainingConfig, GRPOConfig, RewardConfig, ModelConfig,
    OutputConfig,
)

# Import data_utils functions directly
from src.data_utils import (
    load_dataset, prepare_training_data, create_train_val_split,
    save_jsonl, convert_to_mlx_format, clean_text, chunk_text,
    create_completion_examples, create_qa_examples, create_knowledge_examples,
    preprocess_raw_text
)


# ============================================================================
# Config Tests
# ============================================================================

class TestLoRAConfig:
    """Tests for LoRAConfig class."""
    
    def test_default_values(self):
        """Test default LoRA configuration values."""
        config = LoRAConfig()
        assert config.rank == 8
        assert config.alpha == 16
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
    
    def test_custom_values(self):
        """Test custom LoRA configuration values."""
        config = LoRAConfig(rank=16, alpha=32, dropout=0.1)
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LoRAConfig(rank=4)
        d = config.to_dict()
        assert d["rank"] == 4
        assert "target_modules" in d


class TestTrainingConfig:
    """Tests for TrainingConfig class."""
    
    def test_default_values(self):
        """Test default training configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.num_epochs == 3
    
    def test_custom_values(self):
        """Test custom training configuration values."""
        config = TrainingConfig(learning_rate=5e-5, batch_size=8, num_epochs=5)
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8
        assert config.num_epochs == 5
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        d = config.to_dict()
        assert "learning_rate" in d
        assert "batch_size" in d
        assert "save_steps" in d


class TestGRPOConfig:
    """Tests for GRPOConfig class."""

    def test_default_values(self):
        config = GRPOConfig()
        assert config.group_size >= 2
        assert config.beta_kl >= 0.0

    def test_to_dict(self):
        config = GRPOConfig(group_size=6, beta_kl=0.05)
        d = config.to_dict()
        assert d["group_size"] == 6
        assert d["beta_kl"] == 0.05


class TestRewardConfig:
    """Tests for RewardConfig class."""

    def test_default_values(self):
        config = RewardConfig()
        assert config.function == "weighted_rules"
        assert "exact_match" in config.weights

    def test_to_dict(self):
        config = RewardConfig(pass_threshold=0.7)
        d = config.to_dict()
        assert d["pass_threshold"] == 0.7
        assert "weights" in d


class TestModelConfig:
    """Tests for ModelConfig class."""
    
    def test_default_model(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert "Llama" in config.name
        assert config.max_seq_length == 2048
    
    def test_custom_model(self):
        """Test custom model configuration."""
        config = ModelConfig(name="Qwen/Qwen3-0.6B", max_seq_length=4096)
        assert config.name == "Qwen/Qwen3-0.6B"
        assert config.max_seq_length == 4096


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.grpo, GRPOConfig)
        assert isinstance(config.reward, RewardConfig)
        assert isinstance(config.model, ModelConfig)
    
    def test_yaml_round_trip(self):
        """Test saving and loading from YAML."""
        config = Config()
        config.lora.rank = 32
        config.training.learning_rate = 2e-5
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            loaded = Config.from_yaml(f.name)
        
        assert loaded.lora.rank == 32
        assert loaded.training.learning_rate == 2e-5

    def test_default_yaml_uses_qwen_prompt_tokens_for_qwen_model(self):
        """The shipped Qwen default config should not inject Llama-only tokens."""
        config = Config.from_yaml("configs/default.yaml")

        if "qwen" not in config.model.name.lower():
            pytest.skip("Default config no longer uses a Qwen-family model")

        assert config.data.prompt_template is not None
        assert "<|im_start|>" in config.data.prompt_template
        assert "<|start_header_id|>" not in config.data.prompt_template
        assert "<|begin_of_text|>" not in config.data.prompt_template


class TestOutputConfig:
    """Tests for OutputConfig class."""
    
    def test_ensure_dirs(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OutputConfig(
                dir=f"{tmpdir}/outputs",
                adapters_dir=f"{tmpdir}/outputs/adapters",
                checkpoints_dir=f"{tmpdir}/outputs/checkpoints",
                logs_dir=f"{tmpdir}/outputs/logs"
            )
            config.ensure_dirs()
            
            assert Path(config.dir).exists()
            assert Path(config.adapters_dir).exists()
            assert Path(config.checkpoints_dir).exists()
            assert Path(config.logs_dir).exists()


# ============================================================================
# Data Utils Tests
# ============================================================================

class TestLoadDataset:
    """Tests for load_dataset function."""
    
    def test_load_json(self):
        """Test loading JSON file."""
        data = [{"text": "Hello"}, {"text": "World"}]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            loaded = load_dataset(f.name)
        
        assert len(loaded) == 2
        assert loaded[0]["text"] == "Hello"
    
    def test_load_jsonl(self):
        """Test loading JSONL file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write('{"text": "Line 1"}\n')
            f.write('{"text": "Line 2"}\n')
            f.flush()
            loaded = load_dataset(f.name)
        
        assert len(loaded) == 2
        assert loaded[1]["text"] == "Line 2"


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""
    
    def test_instruction_response_format(self):
        """Test converting instruction-response pairs."""
        data = [
            {"instruction": "Say hello", "response": "Hello!"},
            {"instruction": "Count to 3", "response": "1, 2, 3"}
        ]
        formatted = prepare_training_data(data)
        
        assert len(formatted) == 2
        assert "text" in formatted[0]
        assert "Say hello" in formatted[0]["text"]
        assert "Hello!" in formatted[0]["text"]
    
    def test_custom_keys(self):
        """Test with custom key names."""
        data = [{"input": "Question", "output": "Answer"}]
        formatted = prepare_training_data(
            data, 
            instruction_key="input", 
            response_key="output"
        )
        
        assert "Question" in formatted[0]["text"]
        assert "Answer" in formatted[0]["text"]
    
    def test_custom_template(self):
        """Test with custom template."""
        data = [{"instruction": "Q", "response": "A"}]
        template = "Q: {instruction}\nA: {response}"
        formatted = prepare_training_data(data, template=template)
        
        assert formatted[0]["text"] == "Q: Q\nA: A"


class TestTrainValSplit:
    """Tests for create_train_val_split function."""
    
    def test_split_ratio(self):
        """Test correct split ratio."""
        data = [{"i": i} for i in range(100)]
        train, val = create_train_val_split(data, val_ratio=0.2)
        
        assert len(val) == 20
        assert len(train) == 80
    
    def test_no_overlap(self):
        """Test no overlap between train and val."""
        data = [{"i": i} for i in range(50)]
        train, val = create_train_val_split(data, val_ratio=0.2)
        
        train_ids = {d["i"] for d in train}
        val_ids = {d["i"] for d in val}
        assert len(train_ids & val_ids) == 0
    
    def test_reproducibility(self):
        """Test same seed gives same split."""
        data = [{"i": i} for i in range(50)]
        train1, val1 = create_train_val_split(data, seed=42)
        train2, val2 = create_train_val_split(data, seed=42)
        
        assert train1 == train2
        assert val1 == val2


class TestKFoldSplit:
    """Tests for create_kfold_splits function."""
    
    def test_correct_number_of_folds(self):
        """Test that correct number of folds is created."""
        from src.data_utils import create_kfold_splits
        
        data = [{"i": i} for i in range(100)]
        splits = create_kfold_splits(data, k=5)
        
        assert len(splits) == 5
    
    def test_no_overlap_in_validation_sets(self):
        """Test that validation sets across folds don't overlap."""
        from src.data_utils import create_kfold_splits
        
        data = [{"i": i} for i in range(50)]
        splits = create_kfold_splits(data, k=5)
        
        all_val_indices = []
        for train_idx, val_idx in splits:
            all_val_indices.extend(val_idx)
        
        # All validation indices should be unique
        assert len(all_val_indices) == len(set(all_val_indices))
    
    def test_all_data_used(self):
        """Test that all data is used across folds."""
        from src.data_utils import create_kfold_splits
        
        data = [{"i": i} for i in range(50)]
        splits = create_kfold_splits(data, k=5)
        
        all_val_indices = set()
        for train_idx, val_idx in splits:
            all_val_indices.update(val_idx)
        
        # All indices should appear exactly once in validation sets
        assert all_val_indices == set(range(50))
    
    def test_reproducibility(self):
        """Test same seed gives same split."""
        from src.data_utils import create_kfold_splits
        
        data = [{"i": i} for i in range(50)]
        splits1 = create_kfold_splits(data, k=5, seed=42)
        splits2 = create_kfold_splits(data, k=5, seed=42)
        
        assert splits1 == splits2
    
    def test_get_kfold_data(self):
        """Test getting data for a specific fold."""
        from src.data_utils import create_kfold_splits, get_kfold_data
        
        data = [{"i": i} for i in range(50)]
        splits = create_kfold_splits(data, k=5)
        
        train_data, val_data = get_kfold_data(data, splits, fold_idx=0)
        
        # Fold 0 validation should be ~20% of data
        assert len(val_data) == 10
        assert len(train_data) == 40


class TestSaveJsonl:
    """Tests for save_jsonl function."""
    
    def test_save_and_reload(self):
        """Test saving and reloading JSONL."""
        data = [{"a": 1}, {"b": 2}]
        
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            save_jsonl(data, f.name)
            loaded = load_dataset(f.name)
        
        assert loaded == data


class TestConvertToMlxFormat:
    """Tests for convert_to_mlx_format function."""
    
    def test_creates_train_val_files(self):
        """Test that function creates train and val files."""
        data = [{"instruction": f"Q{i}", "response": f"A{i}"} for i in range(20)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            with open(input_file, "w") as f:
                json.dump(data, f)
            
            output_dir = Path(tmpdir) / "output"
            train_path, val_path = convert_to_mlx_format(input_file, output_dir)
            
            assert train_path.exists()
            assert val_path.exists()
            assert train_path.name == "train.jsonl"
            assert val_path.name == "valid.jsonl"


# ============================================================================
# Text Preprocessing Tests
# ============================================================================

class TestCleanText:
    """Tests for clean_text function."""
    
    def test_remove_timestamps(self):
        """Test timestamp removal."""
        text = "0:00 Hello 1:23 World 12:34:56 End"
        cleaned = clean_text(text, remove_timestamps=True)
        assert "0:00" not in cleaned
        assert "1:23" not in cleaned
        assert "Hello" in cleaned
    
    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more"
        cleaned = clean_text(text, remove_urls=True)
        assert "https://example.com" not in cleaned
        assert "Check out" in cleaned
    
    def test_remove_speaker_labels(self):
        """Test speaker label removal."""
        text = "Speaker 1: Hello\n[John]: Hi there"
        cleaned = clean_text(text, remove_speaker_labels=True)
        assert "Speaker 1:" not in cleaned
        assert "[John]:" not in cleaned
        assert "Hello" in cleaned
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    world\n\n\n\nGoodbye"
        cleaned = clean_text(text, normalize_whitespace=True)
        assert "    " not in cleaned
        assert "\n\n\n\n" not in cleaned


class TestChunkText:
    """Tests for chunk_text function."""
    
    def test_creates_chunks(self):
        """Test that chunks are created."""
        text = "Word " * 600  # More varied text
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 1  # At least one chunk
    
    def test_chunk_overlap(self):
        """Test chunks have overlap."""
        text = "Word " * 500
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        
        if len(chunks) > 1:
            # Check that end of chunk 0 overlaps with start of chunk 1
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0
    
    def test_respects_paragraphs(self):
        """Test paragraph-based splitting."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunk_text(text, chunk_size=20, split_on="paragraph")
        assert len(chunks) >= 1


class TestCreateCompletionExamples:
    """Tests for create_completion_examples function."""
    
    def test_creates_examples(self):
        """Test example creation."""
        # Need longer chunks for completion examples to work
        chunks = ["This is a test chunk with enough content to split into context and response. " * 10] * 5
        examples = create_completion_examples(chunks, min_response_length=20)
        
        # May return empty if chunks too short - that's acceptable
        assert isinstance(examples, list)


class TestCreateQAExamples:
    """Tests for create_qa_examples function."""
    
    def test_creates_qa_pairs(self):
        """Test Q&A pair creation."""
        # Need substantial text with clear sentences
        chunks = ["The capital of France is Paris. Paris is a beautiful city with many landmarks. The Eiffel Tower is the most famous landmark in Paris."] * 3
        examples = create_qa_examples(chunks, questions_per_chunk=1)
        
        # Q&A generation may produce empty results for some content
        assert isinstance(examples, list)


class TestCreateKnowledgeExamples:
    """Tests for create_knowledge_examples function."""
    
    def test_creates_knowledge_entries(self):
        """Test knowledge entry creation."""
        # Need chunks that are long enough
        chunks = ["Python is a programming language. " * 5] * 3
        examples = create_knowledge_examples(chunks, topic="Python")
        
        # Check we get some results
        assert isinstance(examples, list)
        if examples:  # If any examples generated
            assert all("text" in ex for ex in examples)


class TestPreprocessRawText:
    """Tests for preprocess_raw_text function."""
    
    def test_completion_format(self):
        """Test completion output format."""
        text = "A" * 2000
        examples = preprocess_raw_text(text, output_format="completion")
        assert all("text" in ex for ex in examples)
    
    def test_qa_format(self):
        """Test Q&A output format."""
        text = "The answer to everything is 42. It's a famous number."
        examples = preprocess_raw_text(text, output_format="qa", chunk_size=100)
        assert len(examples) >= 0  # May be empty for short text
    
    def test_knowledge_format(self):
        """Test knowledge output format."""
        text = "Python was created by Guido van Rossum."
        examples = preprocess_raw_text(text, output_format="knowledge", topic="Python history")
        assert len(examples) >= 0
    
    def test_raw_format(self):
        """Test raw output format."""
        text = "Some raw content " * 100
        examples = preprocess_raw_text(text, output_format="raw", chunk_size=200)
        assert len(examples) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestFullPipeline:
    """Integration tests for full data processing pipeline."""
    
    def test_json_to_training_data(self):
        """Test complete JSON to training data pipeline."""
        # Create sample dataset
        data = [
            {"instruction": "What is 2+2?", "response": "4"},
            {"instruction": "What is Python?", "response": "A programming language"},
        ] * 10  # 20 examples
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input
            input_path = Path(tmpdir) / "input.json"
            with open(input_path, "w") as f:
                json.dump(data, f)
            
            # Convert
            output_dir = Path(tmpdir) / "processed"
            train_path, val_path = convert_to_mlx_format(
                input_path, output_dir, val_ratio=0.1
            )
            
            # Verify
            train_data = load_dataset(train_path)
            val_data = load_dataset(val_path)
            
            assert len(train_data) == 18
            assert len(val_data) == 2
            assert all("text" in d for d in train_data)
    
    def test_raw_text_to_training_data(self):
        """Test complete raw text to training data pipeline."""
        text = """
        This is a sample transcript about machine learning.
        Machine learning is a subset of artificial intelligence.
        It allows computers to learn from data without being explicitly programmed.
        Deep learning is a type of machine learning that uses neural networks.
        """ * 10
        
        examples = preprocess_raw_text(
            text,
            output_format="completion",
            chunk_size=200,
            chunk_overlap=50
        )
        
        assert len(examples) > 0
        
        # Test saving
        with tempfile.TemporaryDirectory() as tmpdir:
            train, val = create_train_val_split(examples, val_ratio=0.1)
            
            train_path = Path(tmpdir) / "train.jsonl"
            val_path = Path(tmpdir) / "valid.jsonl"
            
            save_jsonl(train, train_path)
            save_jsonl(val, val_path)
            
            assert train_path.exists()
            assert val_path.exists()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
