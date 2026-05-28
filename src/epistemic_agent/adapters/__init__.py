"""
Epistemic Agent Adapters Package

Provides integration adapters for various AI agent frameworks.
"""

from .claude_adapter import ClaudeAdapter, EpistemicClaudeClient
from .langchain_adapter import EpistemicCallbackHandler, EpistemicToolWrapper

__all__ = [
    "ClaudeAdapter",
    "EpistemicClaudeClient", 
    "EpistemicCallbackHandler",
    "EpistemicToolWrapper"
]
