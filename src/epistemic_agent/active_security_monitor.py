import math
from typing import Tuple, List
from collections import deque


class ActiveSecurityMonitor:
    """
    Detects hallucinations by calculating calibrated Surprisal (Prediction Error).

    Improvements over raw cosine similarity:
    1. Calibrated probability via temperature-scaled sigmoid:
       P(obs|pred) = σ(β · (sim - τ))
    2. Adaptive threshold using exponential moving average of surprisal
    3. Multi-scale detection: both token-level (Jaccard) and semantic-level

    Research: Surprisal = -ln(P(Observation | Prediction))
    """

    def __init__(
        self,
        beta: float = 8.0,          # Sigmoid sharpness
        tau: float = 0.35,          # Sigmoid midpoint (baseline threshold)
        ema_alpha: float = 0.2,     # EMA smoothing for adaptive threshold
        anomaly_sigma: float = 2.0, # Anomaly = mean + n·std
    ):
        self.beta = beta
        self.tau = tau
        self.ema_alpha = ema_alpha
        self.anomaly_sigma = anomaly_sigma

        # Embedding model
        self._model = None
        self.use_embeddings = False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except ImportError:
            pass

        # Running statistics for adaptive thresholding
        self._surprisal_history: deque = deque(maxlen=50)
        self._mean_surprisal: float = 1.0
        self._var_surprisal: float = 0.5

    def detect_hallucination(self, prediction: str, observation: str) -> Tuple[float, bool]:
        """
        Returns (Surprisal Score, Is_Hallucination).

        Uses calibrated sigmoid probability and adaptive anomaly detection.
        """
        # Multi-scale similarity
        semantic_sim = self._calculate_similarity(prediction, observation)
        token_sim = self._jaccard_similarity(prediction, observation)

        # Use the lower of the two as a conservative estimate
        combined_sim = 0.6 * semantic_sim + 0.4 * token_sim

        # Calibrated probability via sigmoid
        # P = σ(β · (sim - τ)) = 1 / (1 + exp(-β · (sim - τ)))
        prob = self._sigmoid(self.beta * (combined_sim - self.tau))
        prob = max(prob, 1e-8)  # Avoid log(0)

        surprisal = -math.log(prob)

        # Update running statistics
        self._update_running_stats(surprisal)

        # Adaptive anomaly detection: flag if surprisal > μ + k·σ
        std_surprisal = math.sqrt(max(self._var_surprisal, 1e-8))
        threshold = self._mean_surprisal + self.anomaly_sigma * std_surprisal

        # Multi-scale check: flag only if BOTH signals indicate divergence
        semantic_divergent = semantic_sim < 0.35
        token_divergent = token_sim < 0.15

        is_hallucination = (surprisal > threshold) or (semantic_divergent and token_divergent)

        return surprisal, is_hallucination

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)

    def _update_running_stats(self, surprisal: float):
        """
        Update exponential moving average of surprisal statistics.

        μ_new = α·x + (1-α)·μ_old
        σ²_new = α·(x-μ)² + (1-α)·σ²_old
        """
        self._surprisal_history.append(surprisal)

        if len(self._surprisal_history) < 3:
            # Not enough data for adaptive threshold
            self._mean_surprisal = surprisal
            self._var_surprisal = 0.5
            return

        # EMA update
        diff = surprisal - self._mean_surprisal
        self._mean_surprisal += self.ema_alpha * diff
        self._var_surprisal = (
            (1 - self.ema_alpha) * self._var_surprisal
            + self.ema_alpha * diff * diff
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Semantic similarity using sentence embeddings."""
        if not text1 or not text2:
            return 0.0

        if self.use_embeddings and self._model is not None:
            try:
                from sentence_transformers import util
                emb1 = self._model.encode(text1, convert_to_tensor=True)
                emb2 = self._model.encode(text2, convert_to_tensor=True)
                return float(util.pytorch_cos_sim(emb1, emb2)[0][0])
            except Exception:
                pass

        # Fallback to Jaccard
        return self._jaccard_similarity(text1, text2)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Token-level Jaccard similarity."""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def get_diagnostics(self) -> dict:
        """Returns current adaptive threshold state for debugging."""
        std = math.sqrt(max(self._var_surprisal, 1e-8))
        return {
            "mean_surprisal": round(self._mean_surprisal, 4),
            "std_surprisal": round(std, 4),
            "adaptive_threshold": round(self._mean_surprisal + self.anomaly_sigma * std, 4),
            "history_size": len(self._surprisal_history),
            "beta": self.beta,
            "tau": self.tau,
        }
