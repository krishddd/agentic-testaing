"""
Agent Step Data Models for Epistemic Agent

Provides structured data models for capturing agent execution
steps and belief state snapshots.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ActionType(str, Enum):
    """Type of action taken by the agent"""
    EPISTEMIC = "epistemic"  # Information seeking
    PRAGMATIC = "pragmatic"  # Goal achieving


@dataclass
class BeliefSnapshot:
    """Snapshot of agent's belief state at a point in time"""
    file_status_probs: Dict[str, float] = field(default_factory=dict)
    user_intent_probs: Dict[str, float] = field(default_factory=dict)
    risk_level_probs: Dict[str, float] = field(default_factory=dict)
    overall_uncertainty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_status": self.file_status_probs,
            "user_intent": self.user_intent_probs,
            "risk_level": self.risk_level_probs,
            "uncertainty": self.overall_uncertainty
        }
    
    def get_confidence(self) -> float:
        """Calculate overall confidence from belief state"""
        max_probs = []
        if self.file_status_probs:
            max_probs.append(max(self.file_status_probs.values()))
        if self.user_intent_probs:
            max_probs.append(max(self.user_intent_probs.values()))
        if self.risk_level_probs:
            max_probs.append(max(self.risk_level_probs.values()))
        
        if max_probs:
            return sum(max_probs) / len(max_probs)
        return 0.0


@dataclass
class AgentStep:
    """A single step in the agent's execution"""
    loop_number: int
    action_name: str
    action_type: ActionType
    arguments: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    confidence: float = 0.0
    efe_score: float = 0.0
    belief_state: Optional[BeliefSnapshot] = None
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop": self.loop_number,
            "action": self.action_name,
            "type": self.action_type.value,
            "arguments": self.arguments,
            "observation": self.observation[:500] if self.observation else "",
            "confidence": round(self.confidence, 2),
            "efe_score": round(self.efe_score, 2),
            "belief_state": self.belief_state.to_dict() if self.belief_state else None,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning
        }
