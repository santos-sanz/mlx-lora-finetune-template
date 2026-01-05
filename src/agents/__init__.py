# Two-Agent Data Generation Module
"""
This module provides a two-agent system for generating high-quality training data.

- MetaAgent: Analyzes fine-tuning intent and generates specialized prompts
- GeneratorAgent: Uses specialized prompts to create Q&A pairs
"""

from .meta_agent import MetaAgent
from .generator_agent import GeneratorAgent

__all__ = ["MetaAgent", "GeneratorAgent"]
