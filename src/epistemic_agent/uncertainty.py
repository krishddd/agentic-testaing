import math
import re
from typing import List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import ollama
from .config import settings


class UncertaintyEstimator:
    """
    Estimates uncertainty using multiple calibrated methods:

    1. Self-Assessment: Verbalized confidence from the LLM
    2. Self-Consistency: Embedding-based agreement across stochastic samples
    3. Belief Entropy: Direct entropy from the current belief state

    Final combination uses precision-weighted Bayesian fusion:
        combined = (π₁·μ₁ + π₂·μ₂) / (π₁ + π₂)
    where π = 1/σ² is the precision (inverse variance).
    """

    def __init__(self, model_name: str = settings.OLLAMA_MODEL):
        self.model_name = model_name

        # Embedding model for semantic similarity
        self._embedding_model = None
        self._use_embeddings = False
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._use_embeddings = True
        except ImportError:
            pass

        # Running calibration statistics
        self._assessment_history: List[float] = []
        self._consistency_history: List[float] = []

    def _clean_confidence_score(self, response: str) -> float:
        """Extracts a 0.0-1.0 score from a text response."""
        match = re.search(r"(\d+(\.\d+)?)", response)
        if match:
            val = float(match.group(1))
            if val > 1.0 and val <= 10.0:
                return val / 10.0
            if val > 10.0 and val <= 100.0:
                return val / 100.0
            return min(val, 1.0)
        return 0.5

    async def estimate_self_assessment(self, prompt: str, context: str) -> Tuple[float, float]:
        """
        Asks the LLM to rate its own confidence.

        Returns (confidence_score, estimated_precision).
        Precision is derived from historical calibration variance.
        """
        eval_prompt = f"""
        Context: {context}
        User Request: {prompt}

        Analyze if you have ALL necessary facts to safely fulfill this request without hallucinating.
        Output ONLY a confidence score between 0.0 and 1.0.
        """

        try:
            import asyncio
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a calibrated risk assessor. Output only a number.'},
                    {'role': 'user', 'content': eval_prompt}
                ]
            )
            content = response['message']['content']
            score = self._clean_confidence_score(content)

            # Track for calibration
            self._assessment_history.append(score)

            # Precision = 1/σ²  — compute from recent history
            if len(self._assessment_history) >= 3:
                recent = self._assessment_history[-10:]
                if HAS_NUMPY:
                    variance = float(np.var(recent))
                else:
                    mean = sum(recent) / len(recent)
                    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
                precision = 1.0 / max(variance, 0.01)
            else:
                precision = 2.0  # Default moderate precision

            return score, precision

        except Exception as e:
            print(f"Error in self-assessment: {e}")
            return 0.5, 1.0

    async def estimate_self_consistency(
        self, prompt: str, context: str, n_samples: int = 2
    ) -> Tuple[float, float]:
        """
        Multi-sample inference with semantic similarity measurement.

        Instead of naive prefix comparison, uses embedding cosine similarity
        to compute mean pairwise agreement.

        Returns (consistency_score, estimated_precision).
        """
        eval_prompt = f"""
        Context: {context}
        User Request: {prompt}

        Task: Provide a BRIEF 1-sentence summary of the required information to fulfill this request.
        """

        try:
            import asyncio
            tasks = []
            for _ in range(n_samples):
                tasks.append(asyncio.to_thread(
                    ollama.chat,
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': eval_prompt}],
                    options={'temperature': 0.7}
                ))

            responses = await asyncio.gather(*tasks)
            contents = [r['message']['content'].strip() for r in responses]

            # Compute agreement via embeddings or Jaccard fallback
            agreement = self._compute_agreement(contents)

            # Track for calibration
            self._consistency_history.append(agreement)

            # Precision from consistency variance
            if len(self._consistency_history) >= 3:
                recent = self._consistency_history[-10:]
                if HAS_NUMPY:
                    variance = float(np.var(recent))
                else:
                    mean = sum(recent) / len(recent)
                    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
                precision = 1.0 / max(variance, 0.01)
            else:
                precision = 3.0  # Consistency is typically more reliable

            return agreement, precision

        except Exception as e:
            print(f"Error in self-consistency check: {e}")
            return 0.5, 1.0

    def _compute_agreement(self, responses: List[str]) -> float:
        """
        Computes pairwise agreement between response samples.

        Uses embedding cosine similarity if available,
        falls back to Jaccard similarity otherwise.
        """
        if len(responses) < 2:
            return 0.5

        if self._use_embeddings and self._embedding_model is not None:
            try:
                from sentence_transformers import util
                embeddings = self._embedding_model.encode(responses, convert_to_tensor=True)

                # Compute all pairwise cosine similarities
                sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
                n = len(responses)
                pairwise_sims = []
                for i in range(n):
                    for j in range(i + 1, n):
                        pairwise_sims.append(float(sim_matrix[i][j]))

                # Mean pairwise similarity as agreement
                return sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.5

            except Exception:
                pass

        # Fallback: Jaccard similarity
        pairwise_sims = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                set_i = set(responses[i].lower().split())
                set_j = set(responses[j].lower().split())
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                pairwise_sims.append(intersection / union if union > 0 else 0.0)

        return sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.5

    async def get_combined_uncertainty(self, prompt: str, context: str) -> float:
        """Backward-compatible: returns combined confidence only."""
        combined, _ = await self.get_combined_uncertainty_detailed(prompt, context)
        return combined

    async def get_combined_uncertainty_detailed(
        self, prompt: str, context: str, belief_state=None
    ) -> tuple:
        """
        Precision-weighted Bayesian fusion of 3 uncertainty signals:

        combined = (π₁·μ₁ + π₂·μ₂ + π₃·μ₃) / (π₁ + π₂ + π₃)

        Signals:
            μ₁ = self-assessment confidence     (π₁ from calibration variance)
            μ₂ = self-consistency agreement     (π₂ from consistency variance)
            μ₃ = belief-state entropy confidence (π₃ fixed = 1.5)

        Returns (combined_score, details_string).
        """
        assessment, precision_a = await self.estimate_self_assessment(prompt, context)
        consistency, precision_c = await self.estimate_self_consistency(prompt, context)

        # 3rd signal: belief-state entropy → confidence
        # H_max for 4 states = ln(4) ≈ 1.386; normalize entropy to [0,1]
        entropy_confidence = 0.5  # default if no belief state
        precision_e = 1.5        # moderate fixed precision
        if belief_state is not None:
            h = belief_state.overall_uncertainty
            h_max = 1.386 * 3  # ~4.16 max joint entropy (3 factors × 4 states avg)
            entropy_confidence = max(0.0, 1.0 - (h / max(h_max, 1e-8)))

        # Precision-weighted combination
        total_precision = precision_a + precision_c + precision_e
        if total_precision < 1e-8:
            combined = (assessment + consistency + entropy_confidence) / 3.0
        else:
            combined = (
                precision_a * assessment
                + precision_c * consistency
                + precision_e * entropy_confidence
            ) / total_precision

        combined = max(0.0, min(1.0, combined))

        details = (
            f"assess={assessment:.3f}(π={precision_a:.2f}) | "
            f"consist={consistency:.3f}(π={precision_c:.2f}) | "
            f"entropy_conf={entropy_confidence:.3f}(π={precision_e:.2f}) → "
            f"fused={combined:.3f}"
        )

        return combined, details
