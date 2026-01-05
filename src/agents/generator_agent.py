"""
Generator-Agent: Uses specialized prompts to generate high-quality Q&A pairs.

This agent receives a specialized prompt from the Meta-Agent and uses it
to process text chunks, generating training data aligned with the
fine-tuning objectives.
"""

from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def get_openrouter_config() -> Dict[str, str]:
    """Get Open Router configuration from environment variables."""
    return {
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "api_url": os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "qwen/qwen3-0.6b-04-28"),
    }


class GeneratorAgent:
    """
    Generator-Agent that creates Q&A training pairs using specialized prompts.
    
    Uses the prompt from Meta-Agent to generate focused, high-quality
    training data from text chunks.
    """
    
    # Default model for Q&A generation (can be cheaper/faster than Meta-Agent)
    GENERATOR_MODEL = "qwen/qwen3-0.6b-04-28"
    
    def __init__(
        self,
        specialized_prompt: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_workers: int = 10,
    ):
        """
        Initialize the Generator-Agent.
        
        Args:
            specialized_prompt: The prompt from Meta-Agent
            api_key: OpenRouter API key (uses env var if not provided)
            model: Model to use for generation (uses env var if not provided)
            max_workers: Number of parallel workers for processing
        """
        config = get_openrouter_config()
        self.specialized_prompt = specialized_prompt
        self.api_key = api_key or config["api_key"]
        self.api_url = config["api_url"]
        self.model = model or config["model"] or self.GENERATOR_MODEL
        self.max_workers = max_workers
    
    def generate_qa_pairs(
        self,
        chunks: List[str],
        questions_per_chunk: int = 2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from text chunks using the specialized prompt.
        
        Args:
            chunks: List of text chunks to process
            questions_per_chunk: Number of Q&A pairs to generate per chunk
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            List of training examples with 'text' field
        """
        examples = []
        total = len(chunks)
        
        # Prepare arguments for parallel processing
        args_list = [
            (chunk, self.specialized_prompt, self.api_key, self.model, questions_per_chunk)
            for chunk in chunks
        ]
        
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_chunk, args): i 
                for i, args in enumerate(args_list)
            }
            
            for future in as_completed(futures):
                completed += 1
                if progress_callback:
                    progress_callback(completed - 1, total)
                
                try:
                    result = future.result()
                    examples.extend(result)
                except Exception as e:
                    print(f"Warning: Chunk processing failed: {e}")
        
        return examples
    
    def _process_chunk(self, args) -> List[Dict[str, str]]:
        """Process a single chunk to generate Q&A pairs."""
        chunk, specialized_prompt, api_key, model, questions_per_chunk = args
        examples = []
        
        if len(chunk) < 100:
            return examples
        
        chunk_preview = chunk[:2000] if len(chunk) > 2000 else chunk
        
        # Combine specialized prompt with chunk-specific instructions
        full_prompt = f"""{specialized_prompt}

---

TEXT TO PROCESS:
{chunk_preview}

---

Generate {questions_per_chunk} Q&A pairs based on the text above.
Follow the instructions in the prompt carefully.
Format your response as:

1. Q: [Question]
   A: [Answer]

2. Q: [Question]
   A: [Answer]"""
        
        try:
            response = self._call_api(full_prompt, api_key, model)
            examples.extend(self._parse_qa_response(response))
        except Exception as e:
            print(f"Warning: Failed to generate Q&A: {e}")
        
        return examples
    
    def _call_api(
        self,
        prompt: str,
        api_key: str,
        model: str,
        max_tokens: int = 600,
    ) -> str:
        """Call OpenRouter API to generate Q&A pairs."""
        import requests
        
        if not api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mlx-lora-finetune",
            "X-Title": "MLX LoRA Fine-tuning - GeneratorAgent",
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """Parse the API response to extract Q&A pairs."""
        examples = []
        lines = response.split('\n')
        current_q = None
        current_a = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for question patterns
            if line.startswith('Q:') or (len(line) > 0 and line[0].isdigit() and 'Q:' in line):
                # Save previous Q&A if exists
                if current_q and current_a:
                    answer = ' '.join(current_a).strip()
                    if len(answer) > 20 and len(current_q) > 10:
                        text = f"### Instruction:\n{current_q}\n\n### Response:\n{answer}"
                        examples.append({"text": text})
                
                # Extract new question
                if 'Q:' in line:
                    current_q = line.split('Q:', 1)[1].strip()
                else:
                    current_q = line.lstrip('0123456789.-) ').strip()
                current_a = []
            
            elif line.startswith('A:') or (current_q and not (len(line) > 0 and line[0].isdigit() and '.' in line[:3])):
                # Extract answer
                if line.startswith('A:'):
                    current_a.append(line.split('A:', 1)[1].strip())
                elif current_q:
                    current_a.append(line)
        
        # Don't forget the last Q&A pair
        if current_q and current_a:
            answer = ' '.join(current_a).strip()
            if len(answer) > 20 and len(current_q) > 10:
                text = f"### Instruction:\n{current_q}\n\n### Response:\n{answer}"
                examples.append({"text": text})
        
        return examples


def create_agent_qa_examples(
    chunks: List[str],
    intention: str,
    personality: str = "Helpful and informative",
    question_types: Optional[List[str]] = None,
    source_context: str = "",
    api_key: Optional[str] = None,
    meta_model: Optional[str] = None,
    generator_model: Optional[str] = None,
    questions_per_chunk: int = 2,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 10,
) -> List[Dict[str, str]]:
    """
    High-level function to create Q&A examples using the two-agent system.
    
    Args:
        chunks: List of text chunks to process
        intention: Fine-tuning intention/objective
        personality: Target personality/style for responses
        question_types: Types of questions to generate
        source_context: Context about the source content
        api_key: OpenRouter API key
        meta_model: Model for Meta-Agent (smarter model)
        generator_model: Model for Generator-Agent (can be faster/cheaper)
        questions_per_chunk: Q&A pairs per chunk
        progress_callback: Progress callback
        max_workers: Parallel workers
    
    Returns:
        List of training examples
    """
    from .meta_agent import MetaAgent
    
    # Step 1: Meta-Agent generates specialized prompt
    meta_agent = MetaAgent(api_key=api_key, model=meta_model)
    specialized_prompt = meta_agent.generate_specialized_prompt(
        intention=intention,
        personality=personality,
        question_types=question_types,
        source_context=source_context,
    )
    
    # Step 2: Generator-Agent creates Q&A pairs
    generator = GeneratorAgent(
        specialized_prompt=specialized_prompt,
        api_key=api_key,
        model=generator_model,
        max_workers=max_workers,
    )
    
    return generator.generate_qa_pairs(
        chunks=chunks,
        questions_per_chunk=questions_per_chunk,
        progress_callback=progress_callback,
    )
