"""
Constitutional Policy Enforcer (Dynamic Edition)

The 'Judge' of the system. Audits proposed actions against the Security Constitution.

Dynamic capabilities:
  - Uses classify_action_risk() for dynamic safety checks (not hardcoded set)
  - Epistemic actions always allowed (information gathering)
  - Low-risk actions fast-pathed (no LLM call needed)
  - High-risk actions get full LLM constitutional audit
"""

import ollama
from enum import Enum
from typing import Optional
from pydantic import BaseModel
import json
from .security_constitution import SecurityConstitution, default_constitution
from .generative_model import Action, ActionType
from .config import settings
from .escalation_guard import classify_action_risk


class VerdictType(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    NEEDS_CLARIFICATION = "needs_clarification"


class SecurityVerdict(BaseModel):
    verdict: VerdictType
    reasoning: str
    violated_principle: Optional[str] = None


class PolicyEnforcer:
    """
    Constitutional policy enforcement for agent actions.

    Dynamic fast-path:
      - Epistemic actions -> always ALLOWED
      - Actions with risk <= 2 -> ALLOWED (safe by classification)
      - Actions with risk 3-4 -> ALLOWED with note
      - Actions with risk >= 5 -> full LLM constitutional audit
    """

    def __init__(
        self,
        constitution: SecurityConstitution = default_constitution,
        fast_path_threshold: int = 3,
    ):
        self.constitution = constitution
        self.model_name = settings.OLLAMA_MODEL
        self.fast_path_threshold = fast_path_threshold

    def audit_action(self, action: Action, context: str) -> SecurityVerdict:
        """
        Audit an action before execution.

        Uses dynamic risk classification to decide whether a full
        LLM audit is needed, avoiding hardcoded tool lists.
        """
        # FAST PATH 1: Epistemic actions are always allowed
        if action.action_type == ActionType.EPISTEMIC:
            return SecurityVerdict(
                verdict=VerdictType.ALLOWED,
                reasoning="Epistemic verification is permitted to ground truth.",
                violated_principle=None,
            )

        # FAST PATH 2: Dynamic risk classification
        risk = classify_action_risk(action.name)
        if risk <= self.fast_path_threshold:
            return SecurityVerdict(
                verdict=VerdictType.ALLOWED,
                reasoning=f"{action.name} is a low-risk action (risk={risk}/{self.fast_path_threshold}).",
                violated_principle=None,
            )

        # FULL AUDIT: High-risk actions need constitutional review
        return self._llm_audit(action, context, risk)

    def _llm_audit(self, action: Action, context: str, risk: int) -> SecurityVerdict:
        """Full LLM-based constitutional audit for high-risk actions."""
        risk_label = (
            "CRITICAL" if risk >= 8 else
            "HIGH" if risk >= 6 else
            "MODERATE"
        )

        prompt = f"""
        You are the Security Auditor for an AI Agent.
        Your job is to strictly enforce the following Constitution.
        
        {self.constitution.get_prompt_text()}
        
        Current Context:
        {context[:500]}
        
        Proposed Action:
        Name: {action.name}
        Type: {action.action_type}
        Arguments: {action.arguments}
        Description: {action.description}
        Risk Level: {risk_label} ({risk}/10)
        
        Task:
        Determine if this Action violates the Constitution.
        
        CRITICAL RULE FOR EPISTEMIC ACTIONS (web_search, list_files):
        - You MUST ALLOW actions that gather information, even if the query seems wrong.
        - The agent NEEDS to search to confirm facts.
        - Only BLOCK actions that modify state if they are genuinely unsafe.
        
        Output JSON format:
        {{
            "verdict": "allowed" | "blocked" | "needs_clarification",
            "reasoning": "Explanation citing specific principles",
            "violated_principle": "Name of principle or null"
        }}
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You represent the conscience of the AI System.'},
                    {'role': 'user', 'content': prompt}
                ],
                format='json',
                options={'temperature': 0.0}
            )

            content = response['message']['content']
            data = json.loads(content)

            return SecurityVerdict(
                verdict=VerdictType(data['verdict']),
                reasoning=data['reasoning'],
                violated_principle=data.get('violated_principle'),
            )

        except Exception as e:
            print(f"[PolicyEnforcer] Error: {e}")
            # Fail safe
            return SecurityVerdict(
                verdict=VerdictType.BLOCKED,
                reasoning=f"Audit failed due to error: {e}",
                violated_principle="System Integrity",
            )
