from enum import Enum
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
import math
from datetime import datetime


# --- Enums for Discrete State Space ---

class FileStatus(str, Enum):
    EXISTS = "exists"
    DOES_NOT_EXIST = "does_not_exist"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"

class UserIntent(str, Enum):
    DELETE = "delete"
    READ = "read"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"

class RiskLevel(str, Enum):
    SAFE = "safe"
    MODERATE = "moderate"
    HAZARDOUS = "hazardous"

# --- Action Definition ---

class ActionType(str, Enum):
    EPISTEMIC = "epistemic"  # Information seeking
    PRAGMATIC = "pragmatic"  # Goal seeking

class Action(BaseModel):
    name: str
    action_type: ActionType
    arguments: Dict[str, Any] = Field(default_factory=dict)
    description: str

    def __hash__(self):
        return hash((self.name, self.action_type, tuple(sorted(self.arguments.items()))))

# --- Observation Definition ---

class Observation(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    user_input: str
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None
    internal_confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated confidence in current understanding")

    @property
    def content(self) -> str:
        parts = [f"User: {self.user_input}"]
        if self.tool_output:
            parts.append(f"Tool Output: {self.tool_output}")
        if self.tool_error:
            parts.append(f"Tool Error: {self.tool_error}")
        return "\n".join(parts)

# --- Hidden States & Belief State ---

class HiddenState(BaseModel):
    """Represents a single hypothesis about the world state"""
    file_status: FileStatus
    user_intent: UserIntent
    risk_level: RiskLevel


class BeliefState(BaseModel):
    """
    Posterior beliefs over hidden states using Dirichlet concentration parameters.

    Each factor maintains Dirichlet alpha parameters (pseudo-counts).
    The expected probabilities are:  P(k) = α_k / Σα
    The concentration Σα controls confidence — higher = more certain.

    This provides proper Bayesian conjugate updates:
        posterior_α = prior_α + observation_counts
    """
    # Dirichlet concentration parameters (pseudo-counts)
    file_status_alpha: Dict[FileStatus, float] = Field(default_factory=dict)
    user_intent_alpha: Dict[UserIntent, float] = Field(default_factory=dict)
    risk_level_alpha: Dict[RiskLevel, float] = Field(default_factory=dict)

    # Legacy probability access (computed from alphas)
    @property
    def file_status_probs(self) -> Dict[FileStatus, float]:
        return self._alpha_to_probs(self.file_status_alpha)

    @file_status_probs.setter
    def file_status_probs(self, probs: Dict[FileStatus, float]):
        """Backward-compatible setter: converts probabilities to alpha with default concentration."""
        total_alpha = sum(self.file_status_alpha.values()) if self.file_status_alpha else 4.0
        self.file_status_alpha = {k: v * total_alpha for k, v in probs.items()}

    @property
    def user_intent_probs(self) -> Dict[UserIntent, float]:
        return self._alpha_to_probs(self.user_intent_alpha)

    @user_intent_probs.setter
    def user_intent_probs(self, probs: Dict[UserIntent, float]):
        total_alpha = sum(self.user_intent_alpha.values()) if self.user_intent_alpha else 4.0
        self.user_intent_alpha = {k: v * total_alpha for k, v in probs.items()}

    @property
    def risk_level_probs(self) -> Dict[RiskLevel, float]:
        return self._alpha_to_probs(self.risk_level_alpha)

    @risk_level_probs.setter
    def risk_level_probs(self, probs: Dict[RiskLevel, float]):
        total_alpha = sum(self.risk_level_alpha.values()) if self.risk_level_alpha else 3.0
        self.risk_level_alpha = {k: v * total_alpha for k, v in probs.items()}

    # Joint entropy as a scalar uncertainty measure
    @property
    def overall_uncertainty(self) -> float:
        """Joint entropy across all belief factors."""
        h = 0.0
        for probs in [self.file_status_probs, self.user_intent_probs, self.risk_level_probs]:
            for p in probs.values():
                if p > 1e-12:
                    h -= p * math.log(p)
        return h

    @overall_uncertainty.setter
    def overall_uncertainty(self, value: float):
        """No-op setter for backward compatibility."""
        pass

    def _alpha_to_probs(self, alphas: Dict) -> Dict:
        """Convert Dirichlet α to expected probabilities: P(k) = α_k / Σα"""
        if not alphas:
            return {}
        total = sum(alphas.values())
        if total < 1e-12:
            n = len(alphas)
            return {k: 1.0 / n for k in alphas}
        return {k: v / total for k, v in alphas.items()}

    def get_most_likely_state(self) -> HiddenState:
        """Returns the MAP (Maximum A Posteriori) state estimate"""
        return HiddenState(
            file_status=max(self.file_status_alpha, key=self.file_status_alpha.get),
            user_intent=max(self.user_intent_alpha, key=self.user_intent_alpha.get),
            risk_level=max(self.risk_level_alpha, key=self.risk_level_alpha.get)
        )

    def get_concentration(self, factor: str = None) -> float:
        """
        Total concentration (Σα) — measures confidence.
        Higher = more observations incorporated = more certain.
        """
        if factor == "file_status":
            return sum(self.file_status_alpha.values())
        elif factor == "user_intent":
            return sum(self.user_intent_alpha.values())
        elif factor == "risk_level":
            return sum(self.risk_level_alpha.values())
        else:
            return (
                sum(self.file_status_alpha.values())
                + sum(self.user_intent_alpha.values())
                + sum(self.risk_level_alpha.values())
            )

    def bayesian_update(self, factor: str, observation_counts: Dict, learning_rate: float = 1.0):
        """
        Proper Bayesian conjugate update for Categorical-Dirichlet:
            posterior_α_k = prior_α_k + lr · observation_k

        Args:
            factor: 'file_status', 'user_intent', or 'risk_level'
            observation_counts: dict mapping state → observed evidence strength
            learning_rate: scales the update (1.0 = full Bayesian update)
        """
        alpha_map = {
            "file_status": self.file_status_alpha,
            "user_intent": self.user_intent_alpha,
            "risk_level": self.risk_level_alpha
        }
        alphas = alpha_map.get(factor)
        if alphas is None:
            return

        for state, count in observation_counts.items():
            # Handle both enum and string keys
            if isinstance(state, str):
                # Find matching enum
                enum_map = {
                    "file_status": FileStatus,
                    "user_intent": UserIntent,
                    "risk_level": RiskLevel
                }
                try:
                    state = enum_map[factor](state)
                except (ValueError, KeyError):
                    continue

            if state in alphas:
                alphas[state] += learning_rate * max(count, 0.0)

    def compute_variational_free_energy(self, log_likelihood: Dict[str, float] = None) -> float:
        """
        Variational Free Energy:  F = E_q[ln q(s)] - E_q[ln p(o,s)]

        This is a diagnostic signal — lower F means beliefs better explain observations.
        Computed here as the negative entropy (complexity) term.
        If log_likelihood is provided, also includes the accuracy term.
        """
        # Complexity = -H(q) = E_q[ln q(s)]
        complexity = 0.0
        for probs in [self.file_status_probs, self.user_intent_probs, self.risk_level_probs]:
            for p in probs.values():
                if p > 1e-12:
                    complexity += p * math.log(p)

        # Accuracy = E_q[ln p(o|s)]  (if provided)
        accuracy = 0.0
        if log_likelihood:
            for state, ll in log_likelihood.items():
                accuracy += ll  # Already weighted by q in caller

        # F = Complexity - Accuracy  (we want to minimize F)
        return complexity - accuracy

    def normalize(self):
        """
        Ensures all alpha parameters are positive and well-formed.
        With Dirichlet parameters, we don't need to normalize to sum-to-1
        (that's done by _alpha_to_probs), but we ensure no negative values.
        """
        for alphas in [self.file_status_alpha, self.user_intent_alpha, self.risk_level_alpha]:
            for k in alphas:
                alphas[k] = max(alphas[k], 1e-6)


# --- Generative Model Interface ---

class GenerativeModel(BaseModel):
    """
    Generative Model for the Active Inference agent.

    Maintains the belief state using Dirichlet concentration parameters
    for proper Bayesian conjugate updates. The model bridges LLM-based
    approximate inference with principled probabilistic state tracking.

    In a full pymdp implementation, this would hold the A, B, C, D matrices.
    Here the A matrix (observation model) is approximated by the LLM interpreter,
    and the B matrix (transition model) is approximated by the look-ahead module.
    """
    current_belief: BeliefState = Field(default_factory=BeliefState)
    history: List[Observation] = Field(default_factory=list)

    def initialize_belief(self):
        """
        Sets broad Dirichlet priors (high entropy / low concentration).

        Low α values → high uncertainty (uniform-ish distribution).
        Convention: α=1 = uniform prior, α<1 = sparse prior, α>1 = concentrated.
        """
        # Uniform priors for file status (α=1 each → uniform)
        self.current_belief.file_status_alpha = {s: 1.0 for s in FileStatus}

        # Biased prior for intent (mostly unknown at start)
        self.current_belief.user_intent_alpha = {
            UserIntent.DELETE: 0.5,
            UserIntent.READ: 0.5,
            UserIntent.CLARIFY: 0.5,
            UserIntent.UNKNOWN: 2.5,  # Prior belief that intent is unknown
        }

        # Conservative risk prior (lean hazardous until proven safe)
        self.current_belief.risk_level_alpha = {
            RiskLevel.SAFE: 0.6,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HAZARDOUS: 1.5,  # Conservative: assume hazardous
        }

    def update(self, observation: Observation) -> BeliefState:
        """
        Records observation and triggers Bayesian belief update.

        The actual probability updates happen via bayesian_update() calls
        driven by the LLM interpreter's output. This method commits the
        observation to history and returns current belief for chaining.
        """
        self.history.append(observation)
        return self.current_belief
