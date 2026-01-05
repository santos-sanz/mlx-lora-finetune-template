"""
Meta-Agent: Generates specialized prompts based on fine-tuning intent.

This agent analyzes the user's fine-tuning objectives and creates a detailed
prompt that guides the Generator-Agent to produce high-quality Q&A pairs.
"""

from typing import Dict, List, Optional
import os

# Import from parent module's data_utils for API calls
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_openrouter_config() -> Dict[str, str]:
    """Get Open Router configuration from environment variables."""
    return {
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "api_url": os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "qwen/qwen3-0.6b-04-28"),
    }


class MetaAgent:
    """
    Meta-Agent that generates specialized prompts for Q&A generation.
    
    Takes the user's fine-tuning intention and creates a detailed prompt
    that will guide the Generator-Agent to produce relevant, high-quality
    training data.
    """
    
    # Template for generating the meta-prompt
    META_PROMPT_TEMPLATE = """You are an expert at creating training data prompts for fine-tuning language models.

The user wants to fine-tune a model with the following objectives:

## Fine-Tuning Intention
{intention}

## Target Personality/Style
{personality}

## Types of Questions to Generate
{question_types}

## Source Content Summary
{source_context}

---

Based on this information, create a DETAILED PROMPT that will be used to generate Q&A training pairs from text chunks.

The prompt should:
1. Instruct the Q&A generator to focus on questions relevant to the fine-tuning intention
2. Specify the tone and style for answers (matching the target personality)
3. Avoid trivial questions (like asking about addresses, dates, or thank-you messages)
4. Prioritize actionable, practical, and strategic questions
5. Include examples of GOOD and BAD questions for this specific use case

Output ONLY the prompt, nothing else. Format it as instructions that could be given to an AI."""

    # Default prompt for meta-agent model (should be a smart model)
    META_MODEL = "anthropic/claude-3.5-sonnet"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Meta-Agent.
        
        Args:
            api_key: OpenRouter API key (uses env var if not provided)
            model: Model to use for meta-prompt generation (defaults to Claude)
        """
        config = get_openrouter_config()
        self.api_key = api_key or config["api_key"]
        self.api_url = config["api_url"]
        self.model = model or self.META_MODEL
    
    def generate_specialized_prompt(
        self,
        intention: str,
        personality: str = "Helpful and informative",
        question_types: Optional[List[str]] = None,
        source_context: str = "",
    ) -> str:
        """
        Generate a specialized prompt for Q&A generation.
        
        Args:
            intention: What the user wants to achieve with fine-tuning
                      Example: "Train a model that gives business advice like Alex Hormozi"
            personality: The tone/style the model should adopt
                        Example: "Direct, no-nonsense, with concrete numbers and examples"
            question_types: Types of questions to prioritize
                           Example: ["Practical", "Strategic", "How-to"]
            source_context: Brief description of the source content
                           Example: "Books about lead generation and business growth"
        
        Returns:
            A specialized prompt string for the Generator-Agent
        """
        if question_types is None:
            question_types = ["Practical", "Strategic", "Application-based"]
        
        question_types_str = ", ".join(question_types)
        
        # Build the meta-prompt
        meta_prompt = self.META_PROMPT_TEMPLATE.format(
            intention=intention,
            personality=personality,
            question_types=question_types_str,
            source_context=source_context or "General content",
        )
        
        # Call the API to generate the specialized prompt
        try:
            specialized_prompt = self._call_api(meta_prompt)
            return specialized_prompt
        except Exception as e:
            # Fallback to a good default prompt if API fails
            return self._get_fallback_prompt(intention, personality, question_types_str)
    
    def _call_api(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call OpenRouter API to generate the specialized prompt."""
        import requests
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mlx-lora-finetune",
            "X-Title": "MLX LoRA Fine-tuning - MetaAgent",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    def _get_fallback_prompt(
        self,
        intention: str,
        personality: str,
        question_types: str,
    ) -> str:
        """Generate a good fallback prompt if API fails."""
        return f"""You are generating Q&A training pairs for a language model fine-tuning project.

OBJECTIVE: {intention}
STYLE: {personality}
QUESTION TYPES: {question_types}

INSTRUCTIONS:
1. Generate questions that are directly relevant to the objective above
2. Focus on ACTIONABLE, PRACTICAL content that provides real value
3. Answers should match the specified style/personality
4. AVOID trivial questions like:
   - "What is the address mentioned?"
   - "Who is thanked in the text?"
   - "What year was this written?"
   - Simple factual recall without practical application

5. PRIORITIZE questions like:
   - "How can I apply this strategy in my business?"
   - "What are the key steps to achieve [goal]?"
   - "What mistakes should I avoid when [action]?"
   - "How do I measure success with [method]?"

For each text chunk, generate 2-3 high-quality Q&A pairs that capture the core insights and make them actionable.

Format each pair as:
Q: [Question]
A: [Answer in the specified style]"""

    def analyze_source_content(self, text: str, max_chars: int = 5000) -> str:
        """
        Analyze source content to extract context for prompt generation.
        
        Args:
            text: The source text content
            max_chars: Maximum characters to analyze
        
        Returns:
            Brief summary/context of the source content
        """
        sample = text[:max_chars]
        
        analysis_prompt = f"""Analyze this text and provide a 1-2 sentence summary of what it's about and its main themes. Be specific about the domain (business, fitness, tech, etc.):

{sample}

Summary:"""
        
        try:
            return self._call_api(analysis_prompt, max_tokens=150)
        except:
            # Fallback: extract first meaningful paragraph
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            if paragraphs:
                return f"Content about: {paragraphs[0][:200]}..."
            return "General content"
