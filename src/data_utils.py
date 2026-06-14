"""
Data utilities for preparing training and validation datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import random


def load_dataset(path: Union[str, Path], format: str = "auto") -> List[Dict[str, Any]]:
    """Load dataset from file (json or jsonl format)."""
    path = Path(path)
    
    if format == "auto":
        format = "jsonl" if path.suffix == ".jsonl" else "json"
    
    if format == "jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("data", data) if isinstance(data, dict) else data


def prepare_training_data(
    data: List[Dict[str, Any]],
    instruction_key: str = "instruction",
    response_key: str = "response",
    text_key: Optional[str] = None,
    template: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Prepare data for MLX LoRA training format."""
    prepared = []
    
    for item in data:
        if text_key and text_key in item:
            prepared.append({"text": item[text_key]})
        else:
            instruction = item.get(instruction_key, "")
            response = item.get(response_key, "")
            
            if template:
                text = template.format(instruction=instruction, response=response)
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
            prepared.append({"text": text})
    
    return prepared


def create_train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training and validation sets."""
    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)
    
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def create_kfold_splits(
    data: List[Dict[str, Any]],
    k: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Tuple[List[int], List[int]]]:
    """
    Create k-fold cross-validation splits.
    
    Args:
        data: The dataset to split
        k: Number of folds (must be >= 2)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        List of k tuples, each containing (train_indices, val_indices)
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    
    n_samples = len(data)
    if n_samples < k:
        raise ValueError(f"Cannot create {k} folds with only {n_samples} samples")
    
    # Create indices
    indices = list(range(n_samples))
    
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)
    
    # Calculate fold sizes
    fold_size = n_samples // k
    remainder = n_samples % k
    
    # Create folds
    folds = []
    start = 0
    for i in range(k):
        # Distribute remainder across first folds
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end = start + current_fold_size
        folds.append(indices[start:end])
        start = end
    
    # Create train/val splits for each fold
    splits = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        splits.append((train_indices, val_indices))
    
    return splits


def get_kfold_data(
    data: List[Dict[str, Any]],
    splits: List[Tuple[List[int], List[int]]],
    fold_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Get train and validation data for a specific fold.
    
    Args:
        data: The full dataset
        splits: K-fold splits from create_kfold_splits
        fold_idx: Which fold to get (0-indexed)
        
    Returns:
        Tuple of (train_data, val_data) for the specified fold
    """
    train_indices, val_indices = splits[fold_idx]
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    return train_data, val_data


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save data to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_mlx_format(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.1,
    instruction_key: str = "instruction",
    response_key: str = "response",
    template: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Convert dataset to MLX training format with train/val split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    raw_data = load_dataset(input_path)
    prepared_data = prepare_training_data(
        raw_data,
        instruction_key=instruction_key,
        response_key=response_key,
        template=template,
    )
    
    # Split data
    train_data, val_data = create_train_val_split(prepared_data, val_ratio=val_ratio)
    
    # Save files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    
    return train_path, val_path


# ============================================================================
# Text Preprocessing Utilities for Raw Content (Transcripts, Books, etc.)
# ============================================================================

import re


def load_raw_text(path: Union[str, Path]) -> str:
    """Load raw text from file (.txt, .md, etc.)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(
    text: str,
    remove_timestamps: bool = True,
    normalize_whitespace: bool = True,
    remove_urls: bool = False,
    remove_speaker_labels: bool = False,
    min_line_length: int = 0,
) -> str:
    """
    Clean raw text content.
    
    Args:
        text: Raw text to clean
        remove_timestamps: Remove YouTube-style timestamps (00:00, 1:23:45, etc.)
        normalize_whitespace: Collapse multiple spaces/newlines
        remove_urls: Remove HTTP/HTTPS URLs
        remove_speaker_labels: Remove patterns like "Speaker 1:", "[John]:", etc.
        min_line_length: Remove lines shorter than this
    
    Returns:
        Cleaned text
    """
    # Remove timestamps (various formats)
    if remove_timestamps:
        # YouTube format: 0:00, 00:00, 1:23:45
        text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)
        # Bracket format: [00:00], (1:23)
        text = re.sub(r'[\[\(]\d{1,2}:\d{2}(?::\d{2})?[\]\)]', '', text)
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+', '', text)
    
    # Remove speaker labels
    if remove_speaker_labels:
        # Pattern: "Speaker 1:", "[John]:", "NARRATOR:", etc.
        text = re.sub(r'^\s*[\[\(]?[A-Za-z0-9\s]+[\]\)]?\s*:\s*', '', text, flags=re.MULTILINE)
    
    # Filter short lines
    if min_line_length > 0:
        lines = text.split('\n')
        lines = [l for l in lines if len(l.strip()) >= min_line_length]
        text = '\n'.join(lines)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
    
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    split_on: str = "paragraph",
) -> List[str]:
    """
    Split long text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        split_on: How to split - "paragraph", "sentence", or "character"
    
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    if split_on == "paragraph":
        # Split on double newlines (paragraphs)
        segments = re.split(r'\n\s*\n', text)
    elif split_on == "sentence":
        # Split on sentence boundaries
        segments = re.split(r'(?<=[.!?])\s+', text)
    else:
        # Character-level chunking
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        return chunks
    
    # Combine segments into chunks of target size
    chunks = []
    current_chunk = []
    current_size = 0
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
            
        segment_size = len(segment)
        
        if current_size + segment_size <= chunk_size:
            current_chunk.append(segment)
            current_size += segment_size + 2  # +2 for paragraph separator
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [segment]
            current_size = segment_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    # Add overlap by including end of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            overlapped_chunks.append(prev_end + "... " + chunks[i])
        chunks = overlapped_chunks
    
    return chunks


def create_completion_examples(
    chunks: List[str],
    context_ratio: float = 0.6,
    min_response_length: int = 50,
) -> List[Dict[str, str]]:
    """
    Create completion-style training examples from text chunks.
    Given the first part of text, predict the continuation.
    
    Args:
        chunks: List of text chunks
        context_ratio: What fraction of chunk is context vs response
        min_response_length: Minimum response length to include
    
    Returns:
        List of training examples with 'text' field
    """
    examples = []
    
    for chunk in chunks:
        if len(chunk) < min_response_length * 2:
            continue
            
        split_point = int(len(chunk) * context_ratio)
        context = chunk[:split_point].strip()
        response = chunk[split_point:].strip()
        
        if len(response) < min_response_length:
            continue
        
        # Format as instruction-response
        text = f"### Instruction:\nContinue the following text:\n\n{context}\n\n### Response:\n{response}"
        examples.append({"text": text})
    
    return examples


def create_qa_examples(
    chunks: List[str],
    questions_per_chunk: int = 2,
) -> List[Dict[str, str]]:
    """
    Create Q&A style training examples from text chunks.
    Extracts key sentences and creates questions about them.
    
    Args:
        chunks: List of text chunks
        questions_per_chunk: Number of Q&A pairs per chunk
    
    Returns:
        List of training examples
    """
    examples = []
    
    # Question templates
    question_templates = [
        ("What is the main topic discussed in this text?", "summary"),
        ("What are the key points mentioned?", "keypoints"),
        ("Explain the following concept based on the text:", "explain"),
        ("Summarize the information about:", "summarize"),
    ]
    
    for chunk in chunks:
        if len(chunk) < 100:
            continue
        
        # Create "explain this text" type questions
        text = f"### Instruction:\nBased on the following text, answer the question.\n\nText: {chunk[:500]}...\n\nQuestion: What is the main topic discussed?\n\n### Response:\nThe text discusses {chunk[:200].split('.')[0].lower()}..."
        examples.append({"text": text})
        
        # Create "summarize" question
        if len(chunk) > 200:
            summary = chunk[:150].rsplit(' ', 1)[0] + "..."
            text = f"### Instruction:\nSummarize the following text in a few sentences:\n\n{chunk}\n\n### Response:\n{summary}"
            examples.append({"text": text})
    
    return examples


def create_knowledge_examples(
    chunks: List[str],
    topic: str = "the content",
) -> List[Dict[str, str]]:
    """
    Create knowledge-style training examples.
    Format: "Tell me about [topic]" -> content
    
    Args:
        chunks: List of text chunks
        topic: Topic descriptor for the content
    
    Returns:
        List of training examples
    """
    examples = []
    
    prompts = [
        f"Tell me about {topic}",
        f"Explain {topic}",
        f"What do you know about {topic}?",
        f"Give me information about {topic}",
        f"Describe {topic}",
    ]
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < 100:
            continue
        
        prompt = prompts[i % len(prompts)]
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{chunk}"
        examples.append({"text": text})
    
    return examples


def preprocess_raw_text(
    text: str,
    output_format: str = "completion",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    clean_timestamps: bool = True,
    clean_urls: bool = False,
    clean_speaker_labels: bool = False,
    topic: str = "the content",
) -> List[Dict[str, str]]:
    """
    Full preprocessing pipeline for raw text content.
    
    Args:
        text: Raw text content
        output_format: "completion", "qa", "knowledge", or "raw"
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        clean_timestamps: Remove timestamps
        clean_urls: Remove URLs
        clean_speaker_labels: Remove speaker labels
        topic: Topic for knowledge format
    
    Returns:
        List of training examples ready for MLX format
    """
    # Clean the text
    cleaned = clean_text(
        text,
        remove_timestamps=clean_timestamps,
        remove_urls=clean_urls,
        remove_speaker_labels=clean_speaker_labels,
        normalize_whitespace=True,
    )
    
    # Chunk the text
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Generate examples based on format
    if output_format == "completion":
        return create_completion_examples(chunks)
    elif output_format == "qa":
        return create_qa_examples(chunks)
    elif output_format == "knowledge":
        return create_knowledge_examples(chunks, topic=topic)
    elif output_format == "raw":
        # Just return chunks as-is for continued pretraining
        return [{"text": chunk} for chunk in chunks]
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def process_raw_text_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: str = "completion",
    val_ratio: float = 0.1,
    chunk_size: int = 1000,
    **kwargs,
) -> Tuple[Path, Path]:
    """
    Process a raw text file and save as MLX training format.
    
    Args:
        input_path: Path to raw text file
        output_dir: Output directory
        output_format: Training format type
        val_ratio: Validation split ratio
        chunk_size: Chunk size for splitting
        **kwargs: Additional args for preprocess_raw_text
    
    Returns:
        Tuple of (train_path, valid_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load text
    text = load_raw_text(input_path)
    
    # Preprocess
    examples = preprocess_raw_text(
        text,
        output_format=output_format,
        chunk_size=chunk_size,
        **kwargs,
    )
    
    # Split
    train_data, val_data = create_train_val_split(examples, val_ratio=val_ratio)
    
    # Save
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"Processed {len(examples)} examples from raw text")
    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    
    return train_path, val_path


# ============================================================================
# Folder Processing - Process Multiple Text Files
# ============================================================================

def process_folder(
    input_folder: Union[str, Path],
    output_dir: Union[str, Path],
    file_extensions: List[str] = [".txt", ".md"],
    output_format: str = "completion",
    val_ratio: float = 0.1,
    chunk_size: int = 1000,
    clean_timestamps: bool = True,
    clean_urls: bool = False,
    clean_speaker_labels: bool = False,
    topic: str = "the content",
    progress_callback: callable = None,
) -> Tuple[Path, Path]:
    """
    Process all text files in a folder into training data.
    
    Args:
        input_folder: Folder containing text files
        output_dir: Output directory for processed data
        file_extensions: File extensions to process
        output_format: Training format type
        val_ratio: Validation split ratio
        chunk_size: Chunk size for splitting
        clean_timestamps: Remove timestamps
        clean_urls: Remove URLs
        clean_speaker_labels: Remove speaker labels
        topic: Topic for knowledge format
        progress_callback: Optional callback(current, total, filename) for progress
    
    Returns:
        Tuple of (train_path, valid_path)
    """
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    all_files = []
    for ext in file_extensions:
        all_files.extend(input_folder.glob(f"*{ext}"))
        all_files.extend(input_folder.glob(f"**/*{ext}"))  # Recursive
    
    all_files = sorted(set(all_files))  # Remove duplicates and sort
    
    if not all_files:
        raise ValueError(f"No files found with extensions {file_extensions} in {input_folder}")
    
    # Process each file
    all_examples = []
    
    for i, file_path in enumerate(all_files):
        if progress_callback:
            progress_callback(i, len(all_files), file_path.name)
        
        try:
            text = load_raw_text(file_path)
            examples = preprocess_raw_text(
                text,
                output_format=output_format,
                chunk_size=chunk_size,
                chunk_overlap=100,
                clean_timestamps=clean_timestamps,
                clean_urls=clean_urls,
                clean_speaker_labels=clean_speaker_labels,
                topic=topic,
            )
            all_examples.extend(examples)
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue
    
    if not all_examples:
        raise ValueError("No examples generated from any files")
    
    # Shuffle all examples
    random.shuffle(all_examples)
    
    # Split
    train_data, val_data = create_train_val_split(all_examples, val_ratio=val_ratio)
    
    # Save
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"Processed {len(all_files)} files → {len(all_examples)} examples")
    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    
    return train_path, val_path


# ============================================================================
# LLM-Assisted Data Preparation
# ============================================================================

def load_helper_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "mps",
):
    """
    Load a small helper model for intelligent data preparation.
    
    Args:
        model_name: HuggingFace model ID
        device: Device to load on (mps for Apple Silicon)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("transformers and torch required for LLM-assisted prep")
    
    print(f"Loading helper model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    
    return model, tokenizer


def generate_with_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate text using the helper model."""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def create_llm_qa_examples(
    chunks: List[str],
    model,
    tokenizer,
    questions_per_chunk: int = 2,
    progress_callback: callable = None,
) -> List[Dict[str, str]]:
    """
    Create Q&A training examples using an LLM to generate questions.
    
    Args:
        chunks: List of text chunks
        model: Loaded helper model
        tokenizer: Model tokenizer
        questions_per_chunk: Number of Q&A pairs per chunk
        progress_callback: Optional callback(current, total) for progress
    
    Returns:
        List of training examples
    """
    examples = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))
        
        if len(chunk) < 100:
            continue
        
        # Truncate chunk for prompt
        chunk_preview = chunk[:1500] if len(chunk) > 1500 else chunk
        
        # Generate questions
        question_prompt = f"""Based on this text, generate {questions_per_chunk} insightful questions that can be answered from the text.

Text:
{chunk_preview}

Questions (one per line):"""
        
        try:
            questions_text = generate_with_model(model, tokenizer, question_prompt, max_new_tokens=150)
            questions = [q.strip().lstrip("0123456789.-) ") for q in questions_text.split("\n") if q.strip()]
            questions = questions[:questions_per_chunk]
            
            # Generate answer for each question
            for question in questions:
                if not question or len(question) < 10:
                    continue
                
                answer_prompt = f"""Text:
{chunk_preview}

Question: {question}

Answer based only on the text above:"""
                
                answer = generate_with_model(model, tokenizer, answer_prompt, max_new_tokens=200)
                
                if answer and len(answer) > 20:
                    text = f"### Instruction:\n{question}\n\n### Response:\n{answer}"
                    examples.append({"text": text})
        
        except Exception as e:
            print(f"Warning: Failed to generate Q&A for chunk {i}: {e}")
            continue
    
    return examples


def create_llm_summary_examples(
    chunks: List[str],
    model,
    tokenizer,
    progress_callback: callable = None,
) -> List[Dict[str, str]]:
    """
    Create summarization training examples using an LLM.
    
    Args:
        chunks: List of text chunks
        model: Loaded helper model
        tokenizer: Model tokenizer
        progress_callback: Optional callback(current, total) for progress
    
    Returns:
        List of training examples
    """
    examples = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))
        
        if len(chunk) < 200:
            continue
        
        chunk_preview = chunk[:2000] if len(chunk) > 2000 else chunk
        
        summary_prompt = f"""Summarize the following text in 2-3 concise sentences:

Text:
{chunk_preview}

Summary:"""
        
        try:
            summary = generate_with_model(model, tokenizer, summary_prompt, max_new_tokens=150)
            
            if summary and len(summary) > 30:
                text = f"### Instruction:\nSummarize the following text:\n\n{chunk}\n\n### Response:\n{summary}"
                examples.append({"text": text})
        
        except Exception as e:
            print(f"Warning: Failed to generate summary for chunk {i}: {e}")
            continue
    
    return examples


def preprocess_with_llm(
    text: str,
    model,
    tokenizer,
    output_format: str = "qa",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    clean_timestamps: bool = True,
    clean_urls: bool = False,
    clean_speaker_labels: bool = False,
    questions_per_chunk: int = 2,
    progress_callback: callable = None,
) -> List[Dict[str, str]]:
    """
    Full LLM-assisted preprocessing pipeline.
    
    Args:
        text: Raw text content
        model: Loaded helper model
        tokenizer: Model tokenizer
        output_format: "qa" or "summary"
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        clean_timestamps: Remove timestamps
        clean_urls: Remove URLs
        clean_speaker_labels: Remove speaker labels
        questions_per_chunk: Number of Q&A pairs per chunk (for qa format)
        progress_callback: Optional callback for progress
    
    Returns:
        List of training examples
    """
    # Clean the text
    cleaned = clean_text(
        text,
        remove_timestamps=clean_timestamps,
        remove_urls=clean_urls,
        remove_speaker_labels=clean_speaker_labels,
        normalize_whitespace=True,
    )
    
    # Chunk the text
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Generate examples based on format
    if output_format == "qa":
        return create_llm_qa_examples(
            chunks, model, tokenizer,
            questions_per_chunk=questions_per_chunk,
            progress_callback=progress_callback,
        )
    elif output_format == "summary":
        return create_llm_summary_examples(
            chunks, model, tokenizer,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"LLM format must be 'qa' or 'summary', got: {output_format}")


# ============================================================================
# Open Router API for Faster Data Generation
# ============================================================================

import os
import requests

def get_openrouter_config() -> Dict[str, str]:
    """
    Get Open Router configuration from environment variables.
    
    Returns:
        Dict with api_key, api_url, and model
    """
    return {
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "api_url": os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "qwen/qwen3-0.6b-04-28"),
    }


def is_openrouter_configured() -> bool:
    """Check if Open Router API is properly configured."""
    config = get_openrouter_config()
    return bool(config["api_key"] and config["api_key"] != "sk-or-xxxxxxxxxxxxxxxxxxxxxxxx")


def generate_with_openrouter(
    prompt: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Generate text using Open Router API.
    
    Args:
        prompt: The prompt to send to the model
        api_key: Open Router API key (uses env var if not provided)
        api_url: Open Router API URL (uses env var if not provided)
        model: Model to use (uses env var if not provided)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    config = get_openrouter_config()
    api_key = api_key or config["api_key"]
    api_url = api_url or config["api_url"]
    model = model or config["model"]
    
    if not api_key:
        raise ValueError("Open Router API key not configured. Set OPENROUTER_API_KEY in .env")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mlx-lora-finetune",
        "X-Title": "MLX LoRA Fine-tuning",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    response = requests.post(
        f"{api_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    
    if response.status_code != 200:
        raise Exception(f"Open Router API error: {response.status_code} - {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def _process_single_chunk_qa(args) -> List[Dict[str, str]]:
    """Process a single chunk for Q&A - used by thread pool."""
    chunk, api_key, model, questions_per_chunk = args
    examples = []
    
    if len(chunk) < 100:
        return examples
    
    chunk_preview = chunk[:1500] if len(chunk) > 1500 else chunk
    
    # Single prompt that generates complete Q&A pairs
    prompt = f"""Based on this text, generate {questions_per_chunk} question-answer pairs.
Format your response as a numbered list where each item has the question followed by the answer.

Text:
{chunk_preview}

Generate {questions_per_chunk} Q&A pairs in this exact format:
1. Q: [question]
   A: [answer]
2. Q: [question]
   A: [answer]"""
    
    try:
        response = generate_with_openrouter(
            prompt, 
            api_key=api_key, 
            model=model, 
            max_tokens=500
        )
        
        # Parse the response to extract Q&A pairs
        lines = response.split('\n')
        current_q = None
        current_a = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for question patterns
            if line.startswith('Q:') or (line[0].isdigit() and 'Q:' in line):
                # Save previous Q&A if exists
                if current_q and current_a:
                    answer = ' '.join(current_a).strip()
                    if len(answer) > 20:
                        text = f"### Instruction:\n{current_q}\n\n### Response:\n{answer}"
                        examples.append({"text": text})
                
                # Extract new question
                if 'Q:' in line:
                    current_q = line.split('Q:', 1)[1].strip()
                else:
                    current_q = line.lstrip('0123456789.-) ').strip()
                current_a = []
            
            elif line.startswith('A:') or (current_q and not line[0].isdigit()):
                # Extract answer
                if line.startswith('A:'):
                    current_a.append(line.split('A:', 1)[1].strip())
                elif current_q and current_a:  # Continue answer
                    current_a.append(line)
                elif current_q:  # First line of answer without A: prefix
                    current_a.append(line)
        
        # Don't forget the last Q&A pair
        if current_q and current_a:
            answer = ' '.join(current_a).strip()
            if len(answer) > 20:
                text = f"### Instruction:\n{current_q}\n\n### Response:\n{answer}"
                examples.append({"text": text})
    
    except Exception as e:
        print(f"Warning: Failed to generate Q&A: {e}")
    
    return examples


def create_openrouter_qa_examples(
    chunks: List[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    questions_per_chunk: int = 2,
    progress_callback: callable = None,
    max_workers: int = 10,
) -> List[Dict[str, str]]:
    """
    Create Q&A training examples using Open Router API with parallel processing.
    
    Args:
        chunks: List of text chunks
        api_key: Open Router API key
        model: Model to use
        questions_per_chunk: Number of Q&A pairs per chunk
        progress_callback: Optional callback(current, total) for progress
        max_workers: Number of parallel workers (default 10)
    
    Returns:
        List of training examples
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    examples = []
    total = len(chunks)
    
    # Prepare arguments for each chunk
    args_list = [(chunk, api_key, model, questions_per_chunk) for chunk in chunks]
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_process_single_chunk_qa, args): i for i, args in enumerate(args_list)}
        
        # Collect results as they complete
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                # Pass completed-1 so callback receives 0 to total-1 range
                progress_callback(completed - 1, total)
            
            try:
                result = future.result()
                examples.extend(result)
            except Exception as e:
                print(f"Warning: Chunk processing failed: {e}")
    
    return examples


def _process_single_chunk_summary(args) -> Dict[str, str]:
    """Process a single chunk for summary - used by thread pool."""
    chunk, api_key, model = args
    
    if len(chunk) < 200:
        return None
    
    chunk_preview = chunk[:2000] if len(chunk) > 2000 else chunk
    
    summary_prompt = f"""Summarize the following text in 2-3 concise sentences:

Text:
{chunk_preview}

Summary:"""
    
    try:
        summary = generate_with_openrouter(
            summary_prompt, 
            api_key=api_key, 
            model=model, 
            max_tokens=150
        )
        
        if summary and len(summary) > 30:
            text = f"### Instruction:\nSummarize the following text:\n\n{chunk}\n\n### Response:\n{summary}"
            return {"text": text}
    
    except Exception as e:
        print(f"Warning: Failed to generate summary: {e}")
    
    return None


def create_openrouter_summary_examples(
    chunks: List[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    progress_callback: callable = None,
    max_workers: int = 10,
) -> List[Dict[str, str]]:
    """
    Create summarization training examples using Open Router API with parallel processing.
    
    Args:
        chunks: List of text chunks
        api_key: Open Router API key
        model: Model to use
        progress_callback: Optional callback(current, total) for progress
        max_workers: Number of parallel workers (default 10)
    
    Returns:
        List of training examples
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    examples = []
    total = len(chunks)
    
    # Prepare arguments for each chunk
    args_list = [(chunk, api_key, model) for chunk in chunks]
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_process_single_chunk_summary, args): i for i, args in enumerate(args_list)}
        
        # Collect results as they complete
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                # Pass completed-1 so callback receives 0 to total-1 range
                progress_callback(completed - 1, total)
            
            try:
                result = future.result()
                if result:
                    examples.append(result)
            except Exception as e:
                print(f"Warning: Chunk processing failed: {e}")
    
    return examples



def preprocess_with_openrouter(
    text: str,
    output_format: str = "qa",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    clean_timestamps: bool = True,
    clean_urls: bool = False,
    clean_speaker_labels: bool = False,
    questions_per_chunk: int = 2,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    progress_callback: callable = None,
) -> List[Dict[str, str]]:
    """
    Full Open Router preprocessing pipeline.
    
    Args:
        text: Raw text content
        output_format: "qa" or "summary"
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        clean_timestamps: Remove timestamps
        clean_urls: Remove URLs
        clean_speaker_labels: Remove speaker labels
        questions_per_chunk: Number of Q&A pairs per chunk (for qa format)
        api_key: Open Router API key
        model: Model to use
        progress_callback: Optional callback for progress
    
    Returns:
        List of training examples
    """
    # Clean the text
    cleaned = clean_text(
        text,
        remove_timestamps=clean_timestamps,
        remove_urls=clean_urls,
        remove_speaker_labels=clean_speaker_labels,
        normalize_whitespace=True,
    )
    
    # Chunk the text
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Generate examples based on format
    if output_format == "qa":
        return create_openrouter_qa_examples(
            chunks,
            api_key=api_key,
            model=model,
            questions_per_chunk=questions_per_chunk,
            progress_callback=progress_callback,
        )
    elif output_format == "summary":
        return create_openrouter_summary_examples(
            chunks,
            api_key=api_key,
            model=model,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Open Router format must be 'qa' or 'summary', got: {output_format}")


# ============================================================================
# Two-Agent System for High-Quality Data Generation
# ============================================================================

def preprocess_with_agents(
    text: str,
    intention: str,
    personality: str = "Helpful and informative",
    question_types: Optional[List[str]] = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    clean_timestamps: bool = True,
    clean_urls: bool = False,
    clean_speaker_labels: bool = False,
    questions_per_chunk: int = 2,
    api_key: Optional[str] = None,
    meta_model: Optional[str] = None,
    generator_model: Optional[str] = None,
    progress_callback: callable = None,
    max_workers: int = 10,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Two-Agent preprocessing pipeline for high-quality Q&A generation.
    
    Uses a Meta-Agent to analyze the fine-tuning intention and create
    a specialized prompt, then a Generator-Agent to create Q&A pairs
    aligned with the objectives.
    
    Args:
        text: Raw text content
        intention: Fine-tuning intention/objective
                  Example: "Train a model that gives business advice like Alex Hormozi"
        personality: Target personality/style for responses
                    Example: "Direct, practical, with concrete numbers"
        question_types: Types of questions to generate
                       Example: ["Practical", "Strategic", "How-to"]
        chunk_size: Target chunk size (larger for more context)
        chunk_overlap: Overlap between chunks
        clean_timestamps: Remove timestamps
        clean_urls: Remove URLs
        clean_speaker_labels: Remove speaker labels
        questions_per_chunk: Number of Q&A pairs per chunk
        api_key: OpenRouter API key
        meta_model: Model for Meta-Agent (smarter model recommended)
        generator_model: Model for Generator-Agent
        progress_callback: Callback for progress updates
        max_workers: Number of parallel workers
    
    Returns:
        Tuple of (list of training examples, specialized prompt used)
    """
    from src.agents.meta_agent import MetaAgent
    from src.agents.generator_agent import GeneratorAgent
    
    # Clean the text
    cleaned = clean_text(
        text,
        remove_timestamps=clean_timestamps,
        remove_urls=clean_urls,
        remove_speaker_labels=clean_speaker_labels,
        normalize_whitespace=True,
    )
    
    # Chunk the text (larger chunks for more context)
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Step 1: Meta-Agent analyzes content and generates specialized prompt
    meta_agent = MetaAgent(api_key=api_key, model=meta_model)
    
    # Analyze source content for context
    source_context = meta_agent.analyze_source_content(cleaned)
    
    # Generate the specialized prompt
    specialized_prompt = meta_agent.generate_specialized_prompt(
        intention=intention,
        personality=personality,
        question_types=question_types,
        source_context=source_context,
    )
    
    # Step 2: Generator-Agent creates Q&A pairs using the specialized prompt
    generator = GeneratorAgent(
        specialized_prompt=specialized_prompt,
        api_key=api_key,
        model=generator_model,
        max_workers=max_workers,
    )
    
    examples = generator.generate_qa_pairs(
        chunks=chunks,
        questions_per_chunk=questions_per_chunk,
        progress_callback=progress_callback,
    )
    
    return examples, specialized_prompt


def process_with_agents(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    intention: str,
    personality: str = "Helpful and informative",
    question_types: Optional[List[str]] = None,
    val_ratio: float = 0.1,
    chunk_size: int = 1500,
    api_key: Optional[str] = None,
    meta_model: Optional[str] = None,
    generator_model: Optional[str] = None,
    progress_callback: callable = None,
    **kwargs,
) -> Tuple[Path, Path, str]:
    """
    Process a text file using the two-agent system.
    
    Args:
        input_path: Path to raw text file
        output_dir: Output directory
        intention: Fine-tuning intention
        personality: Target personality
        question_types: Types of questions
        val_ratio: Validation split ratio
        chunk_size: Chunk size
        api_key: OpenRouter API key
        meta_model: Meta-Agent model
        generator_model: Generator-Agent model
        progress_callback: Progress callback
        **kwargs: Additional args
    
    Returns:
        Tuple of (train_path, valid_path, specialized_prompt)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load text
    text = load_raw_text(input_path)
    
    # Process with agents
    examples, specialized_prompt = preprocess_with_agents(
        text=text,
        intention=intention,
        personality=personality,
        question_types=question_types,
        chunk_size=chunk_size,
        api_key=api_key,
        meta_model=meta_model,
        generator_model=generator_model,
        progress_callback=progress_callback,
        **kwargs,
    )
    
    # Split
    train_data, val_data = create_train_val_split(examples, val_ratio=val_ratio)
    
    # Save
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"Processed with two-agent system → {len(examples)} examples")
    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    
    return train_path, val_path, specialized_prompt
