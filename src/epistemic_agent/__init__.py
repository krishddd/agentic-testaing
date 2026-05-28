"""
Epistemic Agent Package
=======================

An Active Inference-based agent for AI Safety, implementing:
- POMDP-based Generative Models
- Expected Free Energy (EFE) minimization
- Look-Ahead Simulation
- Information Seeking Loops
- ToolContracts for Safety

New in v0.2.0:
- Universal LLM Gateway (Claude, OpenAI, Ollama, 100+ providers)
- LangChain/CrewAI adapters
"""

__version__ = "0.2.0"

# Core components
from .generative_model import (
    GenerativeModel,
    BeliefState,
    Action,
    ActionType,
    FileStatus,
    UserIntent,
    RiskLevel
)
from .toolgate import ToolGate, ToolContract
from .policy_enforcer import PolicyEnforcer, VerdictType
from .active_security_monitor import ActiveSecurityMonitor

# Security Improvement Modules
from .injection_filter import InjectionFilter
from .belief_monitor import BeliefMonitor
from .exfiltration_guard import ExfiltrationGuard
from .escalation_guard import EscalationGuard
from .cross_validator import CrossValidator
from .adversarial_tests import AdversarialTestSuite

# LLM Gateway
from .llm_gateway import (
    EpistemicLLMGateway,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    get_gateway
)

# Adapters (optional imports)
try:
    from .adapters import (
        ClaudeAdapter,
        EpistemicClaudeClient,
        EpistemicCallbackHandler,
        EpistemicToolWrapper
    )
except ImportError:
    pass  # Adapters require optional dependencies

__all__ = [
    # Core
    "GenerativeModel",
    "BeliefState", 
    "Action",
    "ActionType",
    "FileStatus",
    "UserIntent",
    "RiskLevel",
    "ToolGate",
    "ToolContract",
    "PolicyEnforcer",
    "VerdictType",
    "ActiveSecurityMonitor",
    # Security Modules
    "InjectionFilter",
    "BeliefMonitor",
    "ExfiltrationGuard",
    "EscalationGuard",
    "CrossValidator",
    "AdversarialTestSuite",
    # Gateway
    "EpistemicLLMGateway",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "get_gateway",
    # Adapters
    "ClaudeAdapter",
    "EpistemicClaudeClient",
    "EpistemicCallbackHandler",
    "EpistemicToolWrapper",
]
