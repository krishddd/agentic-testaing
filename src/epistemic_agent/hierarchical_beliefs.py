import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from .generative_model import FileStatus, UserIntent, RiskLevel


@dataclass
class HierarchicalLevel:
    """
    One level in the belief hierarchy.

    Higher levels = more abstract, slower timescale.
    Each level maintains:
        - beliefs: probability distribution over states
        - precision: inverse variance — controls influence weight in updates
        - timescale: how many lower-level steps per update cycle
        - prediction_error_ema: running EMA of prediction errors for precision learning
    """
    level_id: int
    beliefs: Dict[str, float]
    precision: float  # π = 1/σ² — higher = more confident
    timescale: int
    prediction_error_ema: float = 0.0  # Exponential moving average of |ε|²


class HierarchicalBeliefState:
    """
    Multi-level belief hierarchy implementing predictive coding.

    Key improvements over linear interpolation:
    1. Log-space Bayesian fusion for mathematically correct updates
    2. Precision-weighted prediction errors propagated between levels
    3. Adaptive precision learning: π_new = π_old + η·(ε² - 1/π_old)
    4. Proper bottom-up / top-down message passing
    """

    def __init__(self, precision_learning_rate: float = 0.1):
        self.precision_lr = precision_learning_rate

        self.levels = {
            0: HierarchicalLevel(
                level_id=0,
                beliefs={s.value: 1.0 / len(FileStatus) for s in FileStatus},
                precision=1.0,
                timescale=1
            ),
            1: HierarchicalLevel(
                level_id=1,
                beliefs={s.value: 1.0 / len(UserIntent) for s in UserIntent},
                precision=0.8,
                timescale=3
            ),
            2: HierarchicalLevel(
                level_id=2,
                beliefs={s.value: 1.0 / len(RiskLevel) for s in RiskLevel},
                precision=0.6,
                timescale=5
            ),
            3: HierarchicalLevel(
                level_id=3,
                beliefs={'explore': 0.5, 'exploit': 0.5},
                precision=0.5,
                timescale=10
            )
        }

        self.timestep = 0

    def hierarchical_update(
        self,
        observation: Dict,
        level: int = 0
    ):
        """
        Hierarchical belief propagation using precision-weighted prediction errors.

        1. Compute prediction error between levels
        2. Weight errors by precision
        3. Update beliefs using log-space Bayesian fusion
        4. Adapt precision based on prediction error magnitude
        5. Propagate upward if timescale-aligned
        """

        # Bottom-up: update lowest level from observation
        if level == 0 and observation:
            self._update_level_from_observation(0, observation)

        current = self.levels[level]

        # Get top-down prior from level above
        if level < max(self.levels.keys()):
            top_down_prior = self._get_top_down_prior(level + 1, level)
        else:
            top_down_prior = None

        # Get bottom-up likelihood from level below
        if level > 0:
            bottom_up_likelihood = self._get_bottom_up_likelihood(level - 1, level)
        else:
            bottom_up_likelihood = observation if observation else {}

        # Compute prediction error (before update)
        prediction_error = self._compute_prediction_error(
            current.beliefs, bottom_up_likelihood, top_down_prior
        )

        # Log-space Bayesian fusion
        updated_belief = self._log_space_bayesian_fusion(
            current.beliefs,
            top_down_prior,
            bottom_up_likelihood,
            current.precision
        )

        self.levels[level].beliefs = updated_belief

        # Adapt precision based on prediction error
        self._update_precision(level, prediction_error)

        # Propagate upward if timescale aligned
        self.timestep += 1
        if self.timestep % current.timescale == 0:
            if level < max(self.levels.keys()):
                self.hierarchical_update({}, level + 1)

    def _log_space_bayesian_fusion(
        self,
        current_beliefs: Dict[str, float],
        top_down_prior: Optional[Dict[str, float]],
        bottom_up_likelihood: Dict[str, float],
        precision: float
    ) -> Dict[str, float]:
        """
        Proper Bayesian belief update in log-space:

            ln q(s) = π · ln P(o|s) + (1-π) · ln P_top(s) + const

        where π is precision weighting bottom-up vs top-down.
        Softmax normalization ensures valid probability distribution.
        """
        if not current_beliefs:
            return current_beliefs

        log_beliefs = {}
        for state in current_beliefs:
            # Bottom-up: likelihood from observations / lower level
            ll_val = bottom_up_likelihood.get(state, current_beliefs[state]) if bottom_up_likelihood else current_beliefs[state]
            log_likelihood = math.log(max(ll_val, 1e-10))

            # Top-down: prior from higher level
            if top_down_prior and state in top_down_prior:
                td_val = top_down_prior[state]
            else:
                td_val = current_beliefs[state]
            log_prior = math.log(max(td_val, 1e-10))

            # Precision-weighted combination in log-space
            log_beliefs[state] = precision * log_likelihood + (1.0 - precision) * log_prior

        # Softmax normalization (log-sum-exp trick)
        max_log = max(log_beliefs.values())
        exp_values = {k: math.exp(v - max_log) for k, v in log_beliefs.items()}
        total = sum(exp_values.values())

        if total < 1e-12:
            # Fallback to uniform
            n = len(current_beliefs)
            return {k: 1.0 / n for k in current_beliefs}

        return {k: v / total for k, v in exp_values.items()}

    def _compute_prediction_error(
        self,
        current: Dict[str, float],
        bottom_up: Dict[str, float],
        top_down: Optional[Dict[str, float]]
    ) -> float:
        """
        Compute squared prediction error between current beliefs and evidence.

        ε² = Σ_s (bottom_up(s) - current(s))²

        This drives precision learning — large errors → reduce precision
        (less confidence), small errors → increase precision (more confidence).
        """
        if not bottom_up:
            return 0.0

        error_sq = 0.0
        n = 0
        for state in current:
            if state in bottom_up:
                diff = bottom_up[state] - current[state]
                error_sq += diff * diff
                n += 1

        return error_sq / max(n, 1)

    def _update_precision(self, level: int, prediction_error: float):
        """
        Adaptive precision learning:

            π_new = π_old + η · (prediction_error - 1/π_old)

        When prediction_error > 1/π (errors exceed expectation):
            → precision decreases (trust less)
        When prediction_error < 1/π (errors below expectation):
            → precision increases (trust more)

        Clamped to [0.05, 0.99] for stability.
        """
        current = self.levels[level]

        # EMA of prediction error for smoothing
        alpha = 0.3
        current.prediction_error_ema = (
            alpha * prediction_error + (1 - alpha) * current.prediction_error_ema
        )

        # Precision update
        expected_error = 1.0 / max(current.precision, 0.01)
        precision_delta = self.precision_lr * (current.prediction_error_ema - expected_error)
        current.precision += precision_delta

        # Clamp
        current.precision = max(0.05, min(0.99, current.precision))

    def _get_top_down_prior(self, from_level: int, to_level: int) -> Dict:
        """Compute empirical prior from higher level via conditional mappings."""
        higher_beliefs = self.levels[from_level].beliefs
        mapping = {}

        if from_level == 1 and to_level == 0:
            mapping = {
                'delete': {'exists': 0.7, 'ambiguous': 0.2, 'does_not_exist': 0.1},
                'read': {'exists': 0.8, 'unknown': 0.2},
                'clarify': {'ambiguous': 0.6, 'unknown': 0.4},
                'unknown': {s.value: 1.0 / len(FileStatus) for s in FileStatus}
            }
        elif from_level == 2 and to_level == 1:
            mapping = {
                'safe': {'read': 0.6, 'delete': 0.3, 'clarify': 0.1},
                'moderate': {'clarify': 0.5, 'read': 0.4, 'unknown': 0.1},
                'hazardous': {'clarify': 0.7, 'delete': 0.2, 'unknown': 0.1}
            }

        if not mapping:
            return {}

        prior = defaultdict(float)
        for high_state, prob in higher_beliefs.items():
            if high_state in mapping:
                for low_state, cond_prob in mapping[high_state].items():
                    prior[low_state] += prob * cond_prob

        return dict(prior)

    def _get_bottom_up_likelihood(self, from_level: int, to_level: int) -> Dict:
        """Bottom-up message is the lower level's current beliefs."""
        return self.levels[from_level].beliefs

    def _update_level_from_observation(self, level: int, observation: Dict):
        """
        Incorporate observation into level beliefs via Bayesian fusion,
        NOT simple overwrite.
        """
        if not observation:
            return

        current = self.levels[level].beliefs

        # Bayesian fusion: treat observation as likelihood
        updated = self._log_space_bayesian_fusion(
            current_beliefs=current,
            top_down_prior=None,
            bottom_up_likelihood=observation,
            precision=self.levels[level].precision
        )

        self.levels[level].beliefs = updated

    def get_precision_report(self) -> Dict[int, Dict]:
        """Diagnostic: returns precision and prediction error for each level."""
        return {
            level_id: {
                "precision": level.precision,
                "prediction_error_ema": level.prediction_error_ema,
                "entropy": -sum(
                    p * math.log(max(p, 1e-12))
                    for p in level.beliefs.values()
                )
            }
            for level_id, level in self.levels.items()
        }
