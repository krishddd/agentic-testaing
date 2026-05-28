"""
Epistemic Agent API Routes
==========================

Dedicated FastAPI router for Active Inference safety endpoints.
Provides action verification, hallucination detection,
and epistemic agent chat capabilities.
"""

import time
import logging
import traceback
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Router
# ============================================================================

epistemic_router = APIRouter(
    prefix="/api/rag/epistemic",
    tags=["Epistemic Agent - AI Safety"]
)

# ============================================================================
# Request / Response Models
# ============================================================================

class ActionVerifyRequest(BaseModel):
    """Request to verify an action's safety"""
    action_name: str = Field(..., description="Tool name, e.g. 'delete_file', 'web_search'")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Action arguments")
    context: str = Field("", description="Context about why this action is being taken")

class ActionVerifyResponse(BaseModel):
    """Response from action verification"""
    action_name: str
    gate_allowed: bool
    gate_message: str
    policy_verdict: str
    policy_reasoning: str
    risk_level: str
    overall_allowed: bool

class HallucinationCheckRequest(BaseModel):
    """Request to check for hallucination"""
    prediction: str = Field(..., description="What was expected/predicted")
    observation: str = Field(..., description="What was actually observed/received")

class HallucinationCheckResponse(BaseModel):
    """Response from hallucination detection"""
    surprisal: float
    is_hallucination: bool
    threshold: float = 2.0
    explanation: str

class EpistemicChatRequest(BaseModel):
    """Request for epistemic agent chat"""
    query: str = Field(..., description="User query to process through the epistemic agent")
    max_iterations: int = Field(5, ge=1, le=10, description="Max Active Inference loops")

class EpistemicChatResponse(BaseModel):
    """Response from epistemic agent chat"""
    final_answer: str
    agent_type: str = "epistemic_active_inference"
    steps: List[Dict[str, Any]] = []
    belief_evolution: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


# ============================================================================
# Endpoints
# ============================================================================

@epistemic_router.post(
    "/verify-action",
    response_model=ActionVerifyResponse,
    summary="Verify Action Safety"
)
def verify_action(request: ActionVerifyRequest):
    """
    **Verify if an action is safe to execute.**
    
    Checks the action against:
    - **ToolGate**: Precondition contracts (belief state requirements)
    - **PolicyEnforcer**: Constitutional audit (security principles)
    
    Returns whether the action is allowed, blocked, or needs clarification.
    """
    try:
        from src.epistemic_agent.toolgate import ToolGate
        from src.epistemic_agent.policy_enforcer import PolicyEnforcer, VerdictType
        from src.epistemic_agent.generative_model import (
            GenerativeModel, Action, ActionType
        )
        
        tool_gate = ToolGate()
        policy_enforcer = PolicyEnforcer()
        model = GenerativeModel()
        model.initialize_belief()
        
        epistemic_actions = {"web_search", "list_files", "ask_user", "wikipedia"}
        action_type = (
            ActionType.EPISTEMIC if request.action_name in epistemic_actions
            else ActionType.PRAGMATIC
        )
        
        action = Action(
            name=request.action_name,
            action_type=action_type,
            arguments=request.arguments,
            description=request.context
        )
        
        gate_ok, gate_msg = tool_gate.verify_preconditions(action, model.current_belief)
        verdict = policy_enforcer.audit_action(action, request.context)
        
        risk_level = "low"
        if verdict.verdict == VerdictType.BLOCKED:
            risk_level = "high"
        elif verdict.verdict == VerdictType.NEEDS_CLARIFICATION:
            risk_level = "medium"
        
        overall = gate_ok and verdict.verdict == VerdictType.ALLOWED
        
        return ActionVerifyResponse(
            action_name=request.action_name,
            gate_allowed=gate_ok,
            gate_message=gate_msg,
            policy_verdict=verdict.verdict.value,
            policy_reasoning=verdict.reasoning,
            risk_level=risk_level,
            overall_allowed=overall
        )
        
    except Exception as e:
        logger.error(f"Error verifying action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Action verification failed: {str(e)}")


@epistemic_router.post(
    "/detect-hallucination",
    response_model=HallucinationCheckResponse,
    summary="Detect Hallucination"
)
def detect_hallucination(request: HallucinationCheckRequest):
    """
    **Check if an observation is a hallucination.**
    
    Compares a prediction against an observation using semantic similarity.
    High surprisal (prediction error) indicates potential hallucination.
    """
    try:
        from src.epistemic_agent.active_security_monitor import ActiveSecurityMonitor
        
        monitor = ActiveSecurityMonitor()
        surprisal, is_hallucination = monitor.detect_hallucination(
            prediction=request.prediction,
            observation=request.observation
        )
        
        explanation = (
            f"High semantic divergence detected (surprisal={surprisal:.2f}). "
            "The observation significantly differs from the prediction."
            if is_hallucination
            else f"Observation is consistent with prediction (surprisal={surprisal:.2f})."
        )
        
        return HallucinationCheckResponse(
            surprisal=round(surprisal, 4),
            is_hallucination=is_hallucination,
            threshold=2.0,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error in hallucination detection: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Hallucination detection failed: {str(e)}"
        )


@epistemic_router.post(
    "/chat",
    response_model=EpistemicChatResponse,
    summary="Epistemic Agent Chat"
)
async def epistemic_chat(request: EpistemicChatRequest):
    """
    **Process a query through the Active Inference epistemic agent.**
    
    The agent uses:
    - POMDP-based belief tracking
    - Expected Free Energy (EFE) for action selection
    - Constitutional policy enforcement
    - Active hallucination detection
    - Information-seeking loops before pragmatic action
    
    Returns the agent's response with full step-by-step reasoning,
    belief evolution, and safety assessments.
    """
    start_time = time.time()
    
    try:
        from src.epistemic_agent.enhanced_agent import EnhancedEpistemicAgent
        
        agent = EnhancedEpistemicAgent(
            max_iterations=request.max_iterations
        )
        
        steps = []
        belief_snapshots = []
        
        def on_step(step_data):
            """Callback to capture agent steps"""
            if isinstance(step_data, dict):
                steps.append(step_data)
            else:
                steps.append({
                    "loop": getattr(step_data, 'loop_number', 0),
                    "action": getattr(step_data, 'action_name', 'unknown'),
                    "type": getattr(step_data, 'action_type', 'unknown'),
                    "observation": getattr(step_data, 'observation', '')[:300],
                    "confidence": getattr(step_data, 'confidence', 0.0),
                    "efe_score": getattr(step_data, 'efe_score', 0.0),
                })
        
        if hasattr(agent, 'on_step'):
            agent.on_step = on_step
        
        response = await agent.run(request.query)
        
        end_time = time.time()
        
        return EpistemicChatResponse(
            final_answer=response if isinstance(response, str) else str(response),
            agent_type="epistemic_active_inference",
            steps=steps,
            belief_evolution=belief_snapshots,
            metadata={
                "query": request.query,
                "max_iterations": request.max_iterations,
                "total_time_seconds": round(end_time - start_time, 2),
                "total_steps": len(steps),
            }
        )
        
    except Exception as e:
        logger.error(f"Error in epistemic chat: {e}", exc_info=True)
        end_time = time.time()
        return EpistemicChatResponse(
            final_answer=f"Error processing query through epistemic agent: {str(e)}",
            agent_type="epistemic_active_inference",
            steps=[],
            belief_evolution=[],
            metadata={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "total_time_seconds": round(end_time - start_time, 2),
            }
        )
