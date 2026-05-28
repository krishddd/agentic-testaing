import math
import random
from typing import Dict, List, Tuple, Optional
from .generative_model import BeliefState, Action, ActionType, FileStatus, UserIntent, RiskLevel


class FreeEnergyCalculator:
    """
    Calculates Expected Free Energy (G) for action selection using
    proper Active Inference formulations.

    G(π) = Pragmatic Value + Epistemic Value - Repetition Penalty

    Pragmatic Value (Extrinsic):
        -E_q[ln P(o|C)]  — negative expected log-probability of preferred outcomes

    Epistemic Value (Information Gain):
        D_KL(q(s|o,π) || q(s))  — KL divergence between predicted posterior and prior

    Action selection uses softmax with inverse temperature γ:
        P(π) = exp(-γ · G(π)) / Σ_j exp(-γ · G(π_j))
    """

    def __init__(
        self,
        goal_weight: float = 1.0,
        info_weight: float = 2.0,
        gamma: float = 4.0,
        repetition_decay: float = 2.0,
    ):
        self.goal_weight = goal_weight
        self.info_weight = info_weight
        self.gamma = gamma  # Inverse temperature for softmax policy selection
        self.repetition_decay = repetition_decay

    # ── Entropy & Divergence ──────────────────────────────────────

    def calculate_entropy(self, probs: Dict[str, float]) -> float:
        """Shannon entropy  H(P) = -Σ p·ln(p)"""
        entropy = 0.0
        for p in probs.values():
            if p > 1e-12:
                entropy -= p * math.log(p)
        return entropy

    def calculate_kl_divergence(
        self, q: Dict[str, float], p: Dict[str, float]
    ) -> float:
        """
        KL divergence  D_KL(q || p) = Σ q(s)·ln(q(s)/p(s))
        Measures information gained when moving from prior p to posterior q.
        """
        kl = 0.0
        for state in q:
            q_val = max(q.get(state, 1e-12), 1e-12)
            p_val = max(p.get(state, 1e-12), 1e-12)
            kl += q_val * math.log(q_val / p_val)
        return max(kl, 0.0)  # KL is always ≥ 0

    # ── Predicted Posterior (simulated observation) ────────────────

    def _simulate_posterior(
        self, action: Action, current_belief: BeliefState
    ) -> Dict[str, Dict[str, float]]:
        """
        Simulates the expected posterior belief after taking an action.
        Each action type has a DISTINCT information profile so EFE
        can differentiate them.

        Returns dict of factor_name → predicted probability distribution.
        """
        posteriors = {
            "file_status": dict(current_belief.file_status_probs),
            "user_intent": dict(current_belief.user_intent_probs),
            "risk_level": dict(current_belief.risk_level_probs),
        }

        if action.action_type == ActionType.EPISTEMIC:
            if action.name == "list_files":
                # list_files STRONGLY resolves file_status
                # Add directional bias toward EXISTS before sharpening
                # (listing files is expected to confirm existence)
                biased = self._bias_then_sharpen(
                    posteriors["file_status"],
                    bias_key=FileStatus.EXISTS, bias_strength=0.3,
                    temperature=0.15
                )
                posteriors["file_status"] = biased
            elif action.name == "ask_user":
                # ask_user STRONGLY resolves user_intent
                biased = self._bias_then_sharpen(
                    posteriors["user_intent"],
                    bias_key=UserIntent.CLARIFY, bias_strength=0.25,
                    temperature=0.15
                )
                posteriors["user_intent"] = biased
            elif action.name == "web_search":
                # web_search resolves risk and file_status
                posteriors["risk_level"] = self._bias_then_sharpen(
                    posteriors["risk_level"],
                    bias_key=RiskLevel.SAFE, bias_strength=0.2,
                    temperature=0.15
                )
                posteriors["file_status"] = self._sharpen_distribution(
                    posteriors["file_status"], temperature=0.4
                )
            elif action.name == "wikipedia":
                posteriors["risk_level"] = self._sharpen_distribution(
                    posteriors["risk_level"], temperature=0.2
                )
            elif action.name == "read_file":
                posteriors["file_status"] = self._bias_then_sharpen(
                    posteriors["file_status"],
                    bias_key=FileStatus.EXISTS, bias_strength=0.4,
                    temperature=0.1
                )

        return posteriors

    def _sharpen_distribution(
        self, probs: Dict[str, float], temperature: float = 0.5
    ) -> Dict[str, float]:
        """
        Sharpens a distribution by raising to 1/T and re-normalizing.
        Lower temperature → more concentrated (less entropy).
        Simulates the expected entropy reduction from an observation.
        """
        sharpened = {}
        for k, v in probs.items():
            sharpened[k] = max(v, 1e-12) ** (1.0 / max(temperature, 0.01))
        total = sum(sharpened.values())
        return {k: v / total for k, v in sharpened.items()} if total > 0 else probs

    def _bias_then_sharpen(
        self, probs: Dict[str, float], bias_key, bias_strength: float = 0.3,
        temperature: float = 0.5
    ) -> Dict[str, float]:
        """
        Adds directional bias before sharpening to break symmetry.
        Fixes the issue where sharpening uniform distributions has zero effect.

        bias_key: the state expected to gain probability after this action
        bias_strength: how much probability mass to shift toward bias_key
        """
        biased = dict(probs)
        if bias_key in biased:
            # Add bias then re-normalize
            biased[bias_key] += bias_strength
            total = sum(biased.values())
            biased = {k: v / total for k, v in biased.items()}
        return self._sharpen_distribution(biased, temperature)

    # ── Epistemic Value (Information Gain) ────────────────────────

    def calculate_epistemic_value(
        self, action: Action, current_belief: BeliefState
    ) -> float:
        """
        Epistemic value = Expected KL divergence (information gain).

        IG(π) = Σ_factors D_KL(q_predicted(s|π) || q_current(s))

        This measures how much an action is expected to reduce uncertainty
        about the hidden states.
        """
        if action.action_type != ActionType.EPISTEMIC:
            return 0.0

        predicted = self._simulate_posterior(action, current_belief)

        # Weighted sum of KL divergences across all belief factors
        ig_file = self.calculate_kl_divergence(
            predicted["file_status"], dict(current_belief.file_status_probs)
        )
        ig_intent = self.calculate_kl_divergence(
            predicted["user_intent"], dict(current_belief.user_intent_probs)
        )
        ig_risk = self.calculate_kl_divergence(
            predicted["risk_level"], dict(current_belief.risk_level_probs)
        )

        # Weight factors by their current entropy (higher entropy → more valuable to reduce)
        h_file = self.calculate_entropy(dict(current_belief.file_status_probs))
        h_intent = self.calculate_entropy(dict(current_belief.user_intent_probs))
        h_risk = self.calculate_entropy(dict(current_belief.risk_level_probs))
        total_h = h_file + h_intent + h_risk + 1e-8

        weighted_ig = (
            ig_file * (h_file / total_h) * 1.5
            + ig_intent * (h_intent / total_h) * 1.0
            + ig_risk * (h_risk / total_h) * 0.5
        )

        return weighted_ig * self.info_weight

    # ── Pragmatic Value (Extrinsic / Goal-seeking) ────────────────

    def calculate_extrinsic_value(
        self, action: Action, current_belief: BeliefState,
        action_history: List[str] = None,
    ) -> float:
        """
        Pragmatic value = -E_q[ln P(o|C)]
        Measures alignment between predicted outcomes and preferred outcomes.
        Now evidence-aware: answer_user becomes viable after gathering evidence.
        """
        if action.action_type == ActionType.PRAGMATIC:
            risk_safe_prob = current_belief.risk_level_probs.get(RiskLevel.SAFE, 0.0)

            if action.name in ["delete_file", "write_file", "execute_command"]:
                file_exists_prob = current_belief.file_status_probs.get(FileStatus.EXISTS, 0.0)
                risk_moderate_prob = current_belief.risk_level_probs.get(RiskLevel.MODERATE, 0.0)
                combined_safe_risk = risk_safe_prob + risk_moderate_prob  # safe + moderate = acceptable
                
                if combined_safe_risk < 0.5 or file_exists_prob < 0.7:
                    return -15.0  # Not safe enough or file not confirmed
                elif combined_safe_risk < 0.7:
                    return -5.0   # Marginal safety — needs more info
                else:
                    return 10.0 * risk_safe_prob * file_exists_prob

            if action.name == "answer_user":
                total_entropy = (
                    self.calculate_entropy(dict(current_belief.file_status_probs))
                    + self.calculate_entropy(dict(current_belief.user_intent_probs))
                    + self.calculate_entropy(dict(current_belief.risk_level_probs))
                )
                normalized_entropy = total_entropy / 3.87

                # How much evidence have we gathered?
                evidence_count = len(action_history) if action_history else 0

                # Progressive scoring: answer becomes more viable as evidence grows
                risk_safe = current_belief.risk_level_probs.get(RiskLevel.SAFE, 0.0)
                risk_moderate = current_belief.risk_level_probs.get(RiskLevel.MODERATE, 0.0)
                is_low_risk = (risk_safe + risk_moderate) > 0.6
                
                min_evidence = 1 if is_low_risk else 2
                if evidence_count < min_evidence:
                    # Too early to answer — strong penalty
                    return -5.0
                elif normalized_entropy > 0.7:
                    # High uncertainty even after gathering evidence
                    return -2.0 + evidence_count * 0.3
                else:
                    # Enough evidence and manageable uncertainty
                    evidence_bonus = min(evidence_count * 1.5, 6.0)
                    return 4.0 * (1.0 - normalized_entropy) + evidence_bonus

            if action.name == "read_file":
                # Value depends on file existence probability
                file_exists_prob = current_belief.file_status_probs.get(FileStatus.EXISTS, 0.0)
                if file_exists_prob > 0.9:
                    # File confirmed to exist — strongly prefer reading it
                    return 8.0 * file_exists_prob
                elif file_exists_prob > 0.5:
                    return 3.0 * file_exists_prob
                return -2.0

            # ─── New Desktop Tools ────────────────────────────
            if action.name == "move_file":
                file_exists_prob = current_belief.file_status_probs.get(FileStatus.EXISTS, 0.0)
                if file_exists_prob > 0.7:
                    return 5.0 * file_exists_prob
                return -3.0  # Don't move if uncertain file exists

            if action.name == "copy_file":
                file_exists_prob = current_belief.file_status_probs.get(FileStatus.EXISTS, 0.0)
                if file_exists_prob > 0.5:
                    return 4.0 * file_exists_prob
                return -1.0

            if action.name == "write_file":
                # Writing is generally safe (creates new files)
                return 3.0

            if action.name == "get_file_info":
                # Low cost, high info — good epistemic action
                return 2.0

            if action.name == "search_files":
                # Search is epistemic — reduces uncertainty
                return 3.0

            if action.name == "create_directory":
                return 2.5

            if action.name == "get_directory_tree":
                # Very informative, zero risk
                return 3.5

            # Fallback for any unrecognized action
            return 0.0

        # Non-pragmatic actions have zero extrinsic value
        return 0.0

    # ── Combined EFE ──────────────────────────────────────────────

    def calculate_efe(
        self,
        action: Action,
        current_belief: BeliefState,
        action_history: List[str] = None,
    ) -> float:
        """
        Expected Free Energy:  G(π) = Extrinsic + Epistemic - Penalty

        Lower G(π) is BETTER (we minimize EFE).
        But for softmax selection we negate, so higher score = better action.
        """
        epistemic = self.calculate_epistemic_value(action, current_belief)
        extrinsic = self.calculate_extrinsic_value(action, current_belief, action_history)

        score = extrinsic + epistemic

        # Repetition penalty — action-aware decay rates
        if action_history and action.name in action_history:
            repeat_count = action_history.count(action.name)
            if action.action_type == ActionType.EPISTEMIC:
                if action.name == "web_search":
                    # Gentler penalty for web_search: allow 2-3 searches
                    penalty = repeat_count * (self.repetition_decay * 0.5)
                else:
                    # Strong penalty for list_files, ask_user (diminishing returns)
                    penalty = (repeat_count ** 2) * self.repetition_decay
                score -= penalty
            elif action.action_type == ActionType.PRAGMATIC:
                penalty = repeat_count * (self.repetition_decay * 0.75)
                score -= penalty

        return score

    # ── Action Selection ──────────────────────────────────────────

    def select_optimal_action(
        self,
        available_actions: List[Action],
        belief: BeliefState,
        action_history: List[str] = None,
    ) -> Tuple[Action, float]:
        """
        Softmax policy selection with inverse temperature γ.

        P(π) = exp(-γ · G(π)) / Σ_j exp(-γ · G(π_j))

        When γ is high → deterministic (exploitative).
        When γ is low  → stochastic (exploratory).

        Falls back to argmax if all scores identical.
        """
        if not available_actions:
            raise ValueError("No actions available for selection")

        scores = [
            self.calculate_efe(a, belief, action_history) for a in available_actions
        ]

        # Softmax with log-sum-exp trick for numerical stability
        scaled = [self.gamma * s for s in scores]  # Higher score = better
        max_s = max(scaled)
        exp_scores = [math.exp(s - max_s) for s in scaled]
        total = sum(exp_scores)

        if total < 1e-12:
            # All scores identical — pick randomly
            idx = random.randint(0, len(available_actions) - 1)
            return available_actions[idx], scores[idx]

        probs = [e / total for e in exp_scores]

        # Sample action from distribution
        chosen_idx = random.choices(range(len(available_actions)), weights=probs, k=1)[0]

        return available_actions[chosen_idx], scores[chosen_idx]

    # ── Diagnostics ───────────────────────────────────────────────

    def get_action_probabilities(
        self,
        available_actions: List[Action],
        belief: BeliefState,
        action_history: List[str] = None,
    ) -> List[Tuple[Action, float, float]]:
        """Returns (action, efe_score, selection_probability) for all actions."""
        scores = [
            self.calculate_efe(a, belief, action_history) for a in available_actions
        ]
        scaled = [self.gamma * s for s in scores]
        max_s = max(scaled)
        exp_scores = [math.exp(s - max_s) for s in scaled]
        total = sum(exp_scores) + 1e-12
        probs = [e / total for e in exp_scores]

        return [
            (a, s, p)
            for a, s, p in zip(available_actions, scores, probs)
        ]
