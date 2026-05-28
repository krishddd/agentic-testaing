"""
Belief Poisoning Detection

Tracks belief drift across queries within a session.
Detects coordinated attacks that gradually shift beliefs
to make a dangerous action appear safe.

Detection methods:
  1. Drift detection: flags > 30% shift from initial priors
  2. Concentration spike: flags > 3x concentration jump in one loop
  3. Decay mechanism: prevents evidence accumulation attacks
"""

import math
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert when belief drift is detected."""
    is_drifted: bool
    drift_score: float           # 0.0 = no drift, 1.0 = maximum drift
    max_shift: float             # largest single factor shift
    shifted_factors: List[str]   # which factors shifted significantly
    details: str


class BeliefMonitor:
    """
    Monitors belief state changes for signs of belief poisoning.
    
    Tracks:
    - Cumulative drift from initial priors
    - Sudden concentration spikes (evidence flooding)
    - Session-level belief trajectory
    """

    def __init__(
        self,
        drift_threshold: float = 0.30,     # flag if any factor shifts > 30%
        spike_threshold: float = 3.0,      # flag if concentration jumps > 3x
        decay_rate: float = 0.1,           # 10% decay per query
    ):
        self.drift_threshold = drift_threshold
        self.spike_threshold = spike_threshold
        self.decay_rate = decay_rate
        
        self.initial_beliefs: Optional[Dict] = None
        self.previous_concentrations: Optional[Dict] = None
        self.drift_history: List[Dict] = []
        self.alert_count: int = 0
        self.query_count: int = 0

    def set_baseline(self, belief_state) -> None:
        """
        Set the initial belief state as baseline for drift detection.
        Call this at the start of each query/session.
        """
        self.initial_beliefs = {
            "file_status": dict(belief_state.file_status_probs),
            "user_intent": dict(belief_state.user_intent_probs),
            "risk_level": dict(belief_state.risk_level_probs),
        }
        self.previous_concentrations = {
            "file": belief_state.get_concentration("file_status"),
            "intent": belief_state.get_concentration("user_intent"),
            "risk": belief_state.get_concentration("risk_level"),
        }
        self.query_count += 1

    def check_drift(self, current_belief) -> DriftAlert:
        """
        Compare current beliefs against initial baseline.
        Returns DriftAlert if significant drift detected.
        """
        if self.initial_beliefs is None:
            return DriftAlert(
                is_drifted=False, drift_score=0.0, max_shift=0.0,
                shifted_factors=[], details="No baseline set"
            )

        current = {
            "file_status": dict(current_belief.file_status_probs),
            "user_intent": dict(current_belief.user_intent_probs),
            "risk_level": dict(current_belief.risk_level_probs),
        }

        max_shift = 0.0
        total_drift = 0.0
        shifted_factors = []
        factor_count = 0

        for factor_name in ["file_status", "user_intent", "risk_level"]:
            initial = self.initial_beliefs[factor_name]
            curr = current[factor_name]
            
            # Calculate max probability shift for this factor
            for key in initial:
                init_val = initial.get(key, 0.0)
                # Handle enum keys — convert to value for comparison
                key_val = key.value if hasattr(key, 'value') else key
                curr_val = 0.0
                for k, v in curr.items():
                    kv = k.value if hasattr(k, 'value') else k 
                    if kv == key_val:
                        curr_val = v
                        break
                
                shift = abs(curr_val - init_val)
                max_shift = max(max_shift, shift)
                total_drift += shift
                factor_count += 1

                if shift > self.drift_threshold:
                    shifted_factors.append(f"{factor_name}.{key_val}: {init_val:.2f}->{curr_val:.2f}")

        # Normalize drift score
        drift_score = total_drift / max(factor_count, 1)
        is_drifted = max_shift > self.drift_threshold

        if is_drifted:
            self.alert_count += 1
            logger.warning(f"[BeliefMonitor] DRIFT detected: max_shift={max_shift:.2f}, factors={shifted_factors}")

        alert = DriftAlert(
            is_drifted=is_drifted,
            drift_score=round(drift_score, 3),
            max_shift=round(max_shift, 3),
            shifted_factors=shifted_factors,
            details=f"Query #{self.query_count}, drift={drift_score:.3f}, threshold={self.drift_threshold}"
        )

        self.drift_history.append({
            "query": self.query_count,
            "drift_score": alert.drift_score,
            "max_shift": alert.max_shift,
            "is_drifted": alert.is_drifted,
        })

        return alert

    def check_concentration_spike(self, current_belief) -> Optional[Dict]:
        """
        Check if concentration (evidence volume) spiked suspiciously.
        A sudden spike could indicate evidence flooding attack.
        """
        if self.previous_concentrations is None:
            return None

        current_conc = {
            "file": current_belief.get_concentration("file_status"),
            "intent": current_belief.get_concentration("user_intent"),
            "risk": current_belief.get_concentration("risk_level"),
        }

        spikes = {}
        for factor in ["file", "intent", "risk"]:
            prev = self.previous_concentrations[factor]
            curr = current_conc[factor]
            if prev > 0 and curr / prev > self.spike_threshold:
                spikes[factor] = {
                    "previous": round(prev, 2),
                    "current": round(curr, 2),
                    "ratio": round(curr / prev, 2),
                }

        # Update previous for next check
        self.previous_concentrations = current_conc

        if spikes:
            logger.warning(f"[BeliefMonitor] CONCENTRATION SPIKE: {spikes}")
            return spikes
        return None

    def apply_decay(self, belief_state) -> None:
        """
        Apply evidence decay to prevent accumulation attacks.
        Reduces Dirichlet concentration parameters toward priors.
        """
        for factor in ["file_status", "user_intent", "risk_level"]:
            alphas = belief_state._dirichlet_params.get(factor, {})
            for key in alphas:
                # Decay toward 1.0 (uniform prior)
                alphas[key] = 1.0 + (alphas[key] - 1.0) * (1 - self.decay_rate)

    def get_metrics(self) -> Dict:
        """Return belief monitoring metrics."""
        return {
            "total_queries": self.query_count,
            "total_alerts": self.alert_count,
            "drift_history": self.drift_history[-5:],  # Last 5 entries
            "alert_rate": round(self.alert_count / max(self.query_count, 1), 3),
        }
