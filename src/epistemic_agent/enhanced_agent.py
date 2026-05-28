"""
Enhanced Epistemic Agent for Active Inference AI Safety Testing

Pipeline Architecture:
======================
  1. INPUT SECURITY GATE    - Injection filter scan
  2. METRICS INIT           - Initialize epistemic + security metrics
  3. BELIEF INITIALIZATION  - Set priors and baseline
  4. ACTIVE INFERENCE LOOP  - Main perception-action cycle:
     a. UNCERTAINTY CHECK   - EMA-smoothed confidence estimation
     b. ACTION SELECTION    - EFE-based action ranking
     c. CONSTITUTIONAL AUDIT- Policy enforcement
     d. SECURITY GATES      - Escalation + exfiltration guards
     e. EXECUTION           - ToolGate action dispatch
     f. FILE DISCOVERY      - Smart file-matching & auto-execute
     g. OBSERVATION SCAN    - Observation injection filter
     h. HALLUCINATION DET.  - Surprisal-based detection
     i. BELIEF UPDATE       - LLM-interpreted Bayesian update
     j. DRIFT MONITOR       - Belief poisoning detection
     k. METRICS CAPTURE     - Per-loop metrics recording
  5. CONVERGENCE / SYNTHESIS - Final answer generation
"""

import os
import re
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import asyncio

from .config import settings
from .generative_model import RiskLevel
from .generative_model import (
    GenerativeModel, Observation, FileStatus, UserIntent, 
    ActionType, Action, BeliefState
)
from .free_energy import FreeEnergyCalculator
from .uncertainty import UncertaintyEstimator
from .look_ahead import LookAheadModule
from .toolgate import ToolGate
from .mcp_integration import MCPSearchAdapter, SearchResultType
from .test_result import AgentStep, BeliefSnapshot, ActionType as ResultActionType

# Constitutional + Dynamic Security
from .policy_enforcer import PolicyEnforcer, VerdictType
from .active_security_monitor import ActiveSecurityMonitor
from .llm_interpreter import LLMBeliefInterpreter
from .hierarchical_beliefs import HierarchicalBeliefState

# Security Hardening Modules
from .injection_filter import InjectionFilter
from .belief_monitor import BeliefMonitor
from .exfiltration_guard import ExfiltrationGuard
from .escalation_guard import EscalationGuard
from .cross_validator import CrossValidator


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------

class EnhancedEpistemicAgent:
    """
    Enhanced Epistemic Agent with Constitutional Security and Active Inference.

    The agent follows a structured pipeline for every query:
      - Pre-loop: security gate, metrics init, belief reset
      - Loop:     select -> audit -> guard -> execute -> observe -> update
      - Post-loop: synthesise answer from gathered evidence
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        max_iterations: int = None,
        confidence_threshold: float = None,
        on_step: Callable[[AgentStep], None] = None
    ):
        # --- Core Active Inference ---
        self.model = GenerativeModel()
        self.model.initialize_belief()
        self.uncertainty = UncertaintyEstimator()
        self.look_ahead = LookAheadModule(self.uncertainty)
        self.efe_calculator = FreeEnergyCalculator()
        self.tool_gate = ToolGate()
        self.mcp = MCPSearchAdapter()

        # --- Constitutional Security ---
        self.policy_enforcer = PolicyEnforcer()
        self.security_monitor = ActiveSecurityMonitor()
        self.belief_interpreter = LLMBeliefInterpreter()
        self.hierarchical_beliefs = HierarchicalBeliefState()

        # --- Security Hardening ---
        self.injection_filter = InjectionFilter()
        self.belief_monitor = BeliefMonitor()
        self.exfiltration_guard = ExfiltrationGuard()
        self.escalation_guard = EscalationGuard()
        self.cross_validator = CrossValidator()

        # --- LLM-Powered Security (Advanced ML) ---
        self.llm_security = None
        try:
            from .llm_security_analyzer import LLMSecurityAnalyzer
            self.llm_security = LLMSecurityAnalyzer()
        except Exception:
            pass  # Graceful degradation

        # --- Configuration ---
        self.max_iterations = max_iterations or settings.MAX_INFO_LOOP_ITERATIONS
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.on_step = on_step

        # --- Per-query state (reset each run) ---
        self._steps: List[AgentStep] = []
        self._action_history: List[str] = []
        self._ambiguous_files: List[str] = []
        self._epistemic_metrics: Dict[str, Any] = {}

    # ====================================================================
    #  STAGE 1 - INPUT SECURITY GATE
    # ====================================================================

    def _input_security_gate(self, user_input: str):
        """
        Scan user input for prompt-injection attempts.

        Returns:
            scan_result: InjectionScanResult with score, patterns, and verdict
            blocked_response: str if blocked, else None
        """
        scan = self.injection_filter.scan(user_input)
        if scan.is_injected and scan.injection_score >= 0.6:
            print(f"  [X] INPUT INJECTION DETECTED (score={scan.injection_score})")
            blocked = (
                f"I cannot proceed with this request. "
                f"Security Alert: Potential prompt injection detected "
                f"(score={scan.injection_score:.0%}). "
                f"Detected: {', '.join(scan.detected_patterns[:3])}"
            )
            return scan, blocked
        return scan, None

    # ====================================================================
    #  STAGE 2 - METRICS INITIALIZATION
    # ====================================================================

    def _init_metrics(self, input_scan):
        """Initialize the epistemic + security metrics dict for this run."""
        self._epistemic_metrics = {
            "loops": [],
            "final_confidence": 0.0,
            "final_entropy": 0.0,
            "final_vfe": 0.0,
            "total_loops": 0,
            "converged": False,
            "convergence_reason": "",
            "final_beliefs": {},
            "security": {
                "injection_score": round(input_scan.injection_score, 3),
                "injection_blocked": input_scan.is_injected,
                "exfiltration_blocked": False,
                "escalation_detected": False,
                "belief_drift": 0.0,
                "cross_validation_agreement": 1.0,
            },
        }

    # ====================================================================
    #  STAGE 3 - ACTION SELECTION (EFE-based)
    # ====================================================================

    def _select_action(
        self, user_input: str, step: AgentStep
    ) -> Tuple[Action, float, float, float, float]:
        """
        Select the best action via Expected Free Energy minimisation.

        Returns: (best_action, best_score, epistemic_val, extrinsic_val, current_h)
        """
        candidate_actions = self.look_ahead.propose_actions(
            user_input, self.model.current_belief
        )
        best_action, best_score = self.efe_calculator.select_optimal_action(
            candidate_actions, self.model.current_belief, self._action_history
        )

        step.action_name = best_action.name
        step.action_type = (
            ResultActionType.EPISTEMIC
            if best_action.action_type == ActionType.EPISTEMIC
            else ResultActionType.PRAGMATIC
        )
        step.efe_score = best_score

        # --- Display action ranking ---
        all_scores = self.efe_calculator.get_action_probabilities(
            candidate_actions, self.model.current_belief, self._action_history
        )
        policy_scores = getattr(self.look_ahead, 'last_policy_scores', {})
        dh_map = {n: dh for n, dh, _ in policy_scores.get('actions', [])}
        current_h = policy_scores.get('current_h', 0)

        # Deduplicate by name, keep highest EFE
        seen = {}
        for a, s, p in all_scores:
            if a.name not in seen or s > seen[a.name][1]:
                seen[a.name] = (a, s, p)
        ranked = sorted(seen.values(), key=lambda x: -x[1])

        print(f"\n  Action Ranking (H={current_h:.2f}):")
        for a, s, p in ranked:
            star = "  *" if a.name == best_action.name else "   "
            dh = dh_map.get(a.name, 0.0)
            blocked = "  [X]" if s <= -10 else ""
            print(f"{star} {a.name:<14} P={p:>5.1%}  EFE={s:>7.3f}  DH={dh:>+.2f}{blocked}")

        # EFE decomposition
        epistemic_val = self.efe_calculator.calculate_epistemic_value(
            best_action, self.model.current_belief
        )
        extrinsic_val = self.efe_calculator.calculate_extrinsic_value(
            best_action, self.model.current_belief, self._action_history
        )
        print(f"\n  -> {best_action.name}  [info_gain={epistemic_val:.3f} | pragmatic={extrinsic_val:.3f}]")

        return best_action, best_score, epistemic_val, extrinsic_val, current_h

    # ====================================================================
    #  STAGE 4 - CONSTITUTIONAL AUDIT
    # ====================================================================

    def _constitutional_audit(
        self, best_action: Action, context: str, iterations: int
    ) -> Optional[str]:
        """
        Check action against constitutional principles.

        Returns a response string if blocked/paused, else None.
        """
        audit = self.policy_enforcer.audit_action(best_action, context)

        if audit.verdict == VerdictType.BLOCKED:
            print(f"  [X] BLOCKED: {audit.violated_principle} -- {audit.reasoning}")
            self._set_convergence(iterations, f"blocked_by_policy:{audit.violated_principle}")
            return f"I cannot proceed with this action. Security Policy Violation: {audit.reasoning}"

        if audit.verdict == VerdictType.NEEDS_CLARIFICATION:
            print(f"  [||] PAUSE: {audit.reasoning}")
            self._set_convergence(iterations, "needs_clarification_from_policy")
            return f"I need to clarify something before proceeding: {audit.reasoning}"

        return None  # ALLOWED

    # ====================================================================
    #  STAGE 5 - PRE-EXECUTION SECURITY GATES
    # ====================================================================

    def _pre_execution_security(
        self, best_action: Action, step: AgentStep
    ) -> Optional[str]:
        """
        Run escalation guard and exfiltration guard before pragmatic actions.

        Returns observation_str if blocked, else None (continue to execute).
        """
        if best_action.action_type != ActionType.PRAGMATIC:
            return None

        # --- Escalation Guard ---
        esc_alert = self.escalation_guard.record_action(best_action.name)
        if esc_alert.is_escalated:
            print(f"  [!] ESCALATION DETECTED: {esc_alert.details}")
            self._epistemic_metrics["security"]["escalation_detected"] = True
            obs = f"Action Blocked: Privilege escalation detected -- {esc_alert.details}"
            step.observation = obs
            self._emit_step(step)
            return obs  # Signal: skip execution

        # --- Exfiltration Guard (read_file) ---
        if best_action.name == "read_file":
            filepath = best_action.arguments.get("filepath", "")
            is_sensitive, reason = self.exfiltration_guard.is_sensitive_file(filepath)
            if is_sensitive:
                print(f"  [LOCK] EXFILTRATION BLOCKED: {reason}")
                self._epistemic_metrics["security"]["exfiltration_blocked"] = True
                obs = f"Read Blocked: {reason}. This file contains sensitive data."
                step.observation = obs
                self._emit_step(step)
                return obs

        return None

    # ====================================================================
    #  STAGE 6 - ACTION EXECUTION
    # ====================================================================

    async def _execute_action(
        self,
        best_action: Action,
        user_input: str,
        gathered_evidence: List[str],
        step: AgentStep,
        iterations: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute the selected action through ToolGate.

        Handles special cases:
          - answer_user -> synthesize from evidence
          - pragmatic -> verify preconditions first

        Returns:
            (observation_str, final_response)
            - If final_response is not None, the loop should return it.
            - If observation_str is None, caller should `continue`.
        """
        self._action_history.append(best_action.name)

        # --- Special: answer_user ---
        if best_action.name == "answer_user":
            return await self._handle_answer_user(
                user_input, gathered_evidence, step, iterations
            )

        # --- Pragmatic actions need precondition check ---
        if best_action.action_type == ActionType.PRAGMATIC:
            # Security gates already passed (called before this method)
            success, msg = self.tool_gate.verify_preconditions(
                best_action, self.model.current_belief
            )
            if not success:
                return f"Pragmatic Action Blocked: {msg}", None
            return self.tool_gate.execute(best_action), None

        # --- Epistemic actions execute directly ---
        return self.tool_gate.execute(best_action), None

    async def _handle_answer_user(
        self,
        user_input: str,
        gathered_evidence: List[str],
        step: AgentStep,
        iterations: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Handle the answer_user action: synthesize or skip."""
        if not gathered_evidence:
            print(f"  -> answer_user skipped (no evidence yet)")
            step.observation = "No evidence gathered yet -- continuing"
            self._emit_step(step)
            return None, None  # Signal: continue loop

        synthesized = await self._synthesize_answer(user_input, gathered_evidence)

        # --- Exfiltration Guard on output ---
        output_scan = self.exfiltration_guard.scan_output(synthesized)
        if output_scan.is_blocked:
            print(f"  [LOCK] OUTPUT SANITIZED: {output_scan.detected_items}")
            synthesized = output_scan.sanitized_text
            self._epistemic_metrics["security"]["exfiltration_blocked"] = True

        step.observation = synthesized
        print(f"  -> answer_user  [synthesized from {len(gathered_evidence)} evidence items]")
        print(f"  <- {synthesized[:150]}{'...' if len(synthesized) > 150 else ''}")
        self._emit_step(step)
        self._set_convergence(iterations, "answer_synthesized_from_evidence")
        return None, f"Answer: {synthesized}"

    # ====================================================================
    #  STAGE 7 - FILE DISCOVERY & AUTO-EXECUTE
    # ====================================================================

    def _handle_file_discovery(
        self,
        best_action: Action,
        observation_str: str,
        user_input: str,
        step: AgentStep,
        context: str,
        gathered_evidence: List[str],
        iterations: int,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Handle file listing results: update beliefs, auto-execute if appropriate.

        Returns:
            (updated_observation, updated_context, final_response_or_None)
        """
        if best_action.name != "list_files":
            return observation_str, context, None

        # --- Ambiguous match ---
        if "ambiguous" in observation_str.lower():
            return self._handle_ambiguous_files(observation_str, context)

        # --- Files found ---
        if "found" not in observation_str.lower():
            return observation_str, context, None

        file_list = re.findall(r"'([^']+)'", observation_str)
        user_lower = user_input.lower()

        is_delete = any(w in user_lower for w in ['delete', 'remove', 'erase', 'trash'])
        is_read = any(w in user_lower for w in ['read', 'open', 'show me', 'cat', 'display', 'contents of'])
        is_list = any(w in user_lower for w in ['list', 'list all', 'find all', 'show all'])

        # Update beliefs based on discovery
        self._update_file_beliefs(file_list, is_delete, is_read, is_list)

        # Auto-execute if single clear intent
        if file_list and (is_delete or is_read) and not is_list:
            result = self._auto_execute_file_action(
                file_list, user_input, observation_str,
                is_delete, is_read, step, iterations
            )
            if result:
                return observation_str, context, result

        return observation_str, context, None

    def _handle_ambiguous_files(self, observation_str: str, context: str):
        """Handle the case where list_files returns ambiguous results."""
        print(f"  [!] AMBIGUOUS -- Multiple files match")
        observation_str += "\n[!] AMBIGUOUS: Multiple files match. User must specify exact filename."
        found_files = re.findall(r"'([^']+)'", observation_str)

        self.model.current_belief.file_status_probs = {
            FileStatus.EXISTS: 0.20,
            FileStatus.DOES_NOT_EXIST: 0.02,
            FileStatus.AMBIGUOUS: 0.70,
            FileStatus.UNKNOWN: 0.08,
        }
        self.model.current_belief.bayesian_update('risk_level', {
            RiskLevel.SAFE: 0.30, RiskLevel.MODERATE: 0.50, RiskLevel.HAZARDOUS: 0.20
        })
        self.model.current_belief.bayesian_update('file_status', {FileStatus.AMBIGUOUS: 3.0})
        self.model.current_belief.bayesian_update('user_intent', {UserIntent.DELETE: 2.0})
        self._ambiguous_files = found_files

        return observation_str, context + f"\nAMBIGUOUS FILES FOUND: {found_files}", None

    def _update_file_beliefs(self, file_list, is_delete, is_read, is_list):
        """Update Bayesian beliefs after file discovery."""
        if len(file_list) == 1:
            print(f"  [OK] EXACT MATCH: {file_list[0]}")
            self.model.current_belief.file_status_probs = {
                FileStatus.EXISTS: 0.99,
                FileStatus.DOES_NOT_EXIST: 0.003,
                FileStatus.AMBIGUOUS: 0.004,
                FileStatus.UNKNOWN: 0.003,
            }
        elif len(file_list) > 1:
            print(f"  [OK] FOUND {len(file_list)} files")
            self.model.current_belief.file_status_probs = {
                FileStatus.EXISTS: 0.95,
                FileStatus.DOES_NOT_EXIST: 0.01,
                FileStatus.AMBIGUOUS: 0.02,
                FileStatus.UNKNOWN: 0.02,
            }
            if is_delete:
                self.model.current_belief.bayesian_update('user_intent', {
                    UserIntent.DELETE: 3.0, UserIntent.UNKNOWN: 0.5
                })
                self.model.current_belief.bayesian_update('risk_level', {
                    RiskLevel.MODERATE: 2.0, RiskLevel.SAFE: 1.0
                })
            elif is_list:
                self.model.current_belief.bayesian_update('user_intent', {
                    UserIntent.READ: 3.0, UserIntent.UNKNOWN: 0.5
                })
                self.model.current_belief.bayesian_update('risk_level', {RiskLevel.SAFE: 3.0})
            else:
                self.model.current_belief.bayesian_update('user_intent', {
                    UserIntent.READ: 2.0, UserIntent.UNKNOWN: 0.5
                })
                self.model.current_belief.bayesian_update('risk_level', {RiskLevel.SAFE: 2.0})

    def _auto_execute_file_action(
        self, file_list, user_input, observation_str,
        is_delete, is_read, step, iterations
    ) -> Optional[str]:
        """
        Auto-execute a file action when a clear match + intent is found.

        Returns a final response string if executed, else None.
        """
        target = self.look_ahead._extract_filename(user_input)
        best_match = self._find_best_file_match(file_list, target)
        if not best_match:
            return None

        suggested_path = self.look_ahead._extract_path(user_input)
        full_path = (
            os.path.join(suggested_path, best_match)
            if suggested_path != "." else best_match
        )

        if is_delete:
            return self._auto_delete(full_path, user_input, observation_str, step, iterations)
        elif is_read and len(file_list) == 1:
            return self._auto_read(full_path, observation_str, step, iterations)
        return None

    def _find_best_file_match(self, file_list, target) -> Optional[str]:
        """Find best matching file from discovered list."""
        if not target:
            return None
        for f in file_list:
            if f == target or os.path.splitext(f)[0] == target:
                return f
        for f in file_list:
            if f.lower().startswith(target.lower()):
                return f
        return None

    def _auto_delete(self, full_path, user_input, observation_str, step, iterations) -> Optional[str]:
        """Auto-execute delete with cross-validation."""
        print(f"  >> AUTO-EXECUTE: delete_file -> {full_path}")
        delete_action = Action(
            name="delete_file", action_type=ActionType.PRAGMATIC,
            arguments={"filepath": full_path}, description=f"Delete {full_path}"
        )
        success, msg = self.tool_gate.verify_preconditions(delete_action, self.model.current_belief)
        if not success:
            print(f"  [!] Delete precondition failed: {msg}")
            return None

        # Cross-validation for destructive action
        cv_result = self.cross_validator.validate(
            action_name="delete_file",
            action_args={"filepath": full_path},
            user_input=user_input,
            context=observation_str[:300],
        )
        self._epistemic_metrics["security"]["cross_validation_agreement"] = cv_result.agreement_score
        if not cv_result.is_agreed:
            print(f"  [!] CROSS-VALIDATOR DISAGREES: {cv_result.validator_reasoning}")
            step.observation = f"{observation_str}\n-> Cross-validation failed: {cv_result.validator_reasoning}"
            self._emit_step(step)
            return None  # Will ask_user in next loop

        delete_result = self.tool_gate.execute(delete_action)
        step.observation = f"{observation_str}\n-> EXECUTED: {delete_result}"
        self._emit_step(step)
        self._set_convergence(iterations, "pragmatic_action_auto_executed")
        return f"Action completed: {delete_result}"

    def _auto_read(self, full_path, observation_str, step, iterations) -> Optional[str]:
        """Auto-execute read_file when exact match found."""
        print(f"  >> AUTO-EXECUTE: read_file -> {full_path}")
        read_action = Action(
            name="read_file", action_type=ActionType.PRAGMATIC,
            arguments={"filepath": full_path}, description=f"Read {full_path}"
        )
        success, msg = self.tool_gate.verify_preconditions(read_action, self.model.current_belief)
        if not success:
            return None

        read_result = self.tool_gate.execute(read_action)
        step.observation = f"{observation_str}\n-> EXECUTED: {read_result}"
        self._emit_step(step)
        self._set_convergence(iterations, "pragmatic_action_auto_executed")
        return f"Action completed: {read_result}"

    # ====================================================================
    #  STAGE 8 - POST-EXECUTION ANALYSIS
    # ====================================================================

    def _scan_observation_injection(self, observation_str: str) -> str:
        """Scan observation for injected content; sanitize if detected."""
        obs_scan = self.injection_filter.scan(observation_str)
        if obs_scan.is_injected:
            print(f"  [!] INJECTION in observation (score={obs_scan.injection_score}): {obs_scan.detected_patterns[:2]}")
            self._epistemic_metrics["security"]["injection_score"] = max(
                self._epistemic_metrics["security"]["injection_score"],
                obs_scan.injection_score,
            )
            return obs_scan.sanitized_text
        return observation_str

    def _detect_hallucination(
        self, best_action: Action, user_input: str, observation_str: str
    ) -> Tuple[float, bool]:
        """Run surprisal-based hallucination detection."""
        surprisal, is_hallucination = self.security_monitor.detect_hallucination(
            prediction=f"Result for {best_action.name} relevant to {user_input}",
            observation=observation_str,
        )
        return surprisal, is_hallucination

    async def _update_beliefs(
        self, user_input: str, observation_str: str
    ) -> Dict:
        """Run LLM belief interpreter and apply Bayesian updates."""
        interpretation = await self.belief_interpreter.interpret_observation(
            user_input, observation_str,
            self.model.current_belief, self._action_history
        )
        self._apply_belief_update(interpretation['belief_update'])
        self.hierarchical_beliefs.hierarchical_update(
            interpretation['belief_update'].get('file_status', {}), level=0
        )
        self.hierarchical_beliefs.hierarchical_update(
            interpretation['belief_update'].get('user_intent', {}), level=1
        )
        return interpretation

    def _check_belief_drift(self):
        """Monitor for belief poisoning attacks."""
        drift_alert = self.belief_monitor.check_drift(self.model.current_belief)
        conc_spike = self.belief_monitor.check_concentration_spike(self.model.current_belief)

        if drift_alert.is_drifted:
            print(f"  [!] BELIEF DRIFT: max_shift={drift_alert.max_shift:.2f}, factors={drift_alert.shifted_factors}")
            self._epistemic_metrics["security"]["belief_drift"] = drift_alert.drift_score
        if conc_spike:
            print(f"  [!] CONCENTRATION SPIKE: {conc_spike}")

    # ====================================================================
    #  STAGE 9 - METRICS CAPTURE
    # ====================================================================

    def _capture_loop_metrics(
        self, iterations: int, best_action: Action, best_score: float,
        confidence: float, current_h: float,
        epistemic_val: float, extrinsic_val: float,
        surprisal: float, is_hallucination: bool,
    ):
        """Record per-loop epistemic metrics."""
        belief = self.model.current_belief
        conc_file = belief.get_concentration('file_status')
        conc_intent = belief.get_concentration('user_intent')
        conc_risk = belief.get_concentration('risk_level')
        vfe = belief.compute_variational_free_energy()

        def _dominant(probs):
            best_k, best_v = max(probs.items(), key=lambda x: x[1])
            return f"{best_k.value}({best_v:.0%})"

        surprisal_status = f"S={surprisal:.2f}{'  [!] HALLUCINATION' if is_hallucination else ' [OK]'}"
        print(f"\n  Beliefs: file={_dominant(belief.file_status_probs)} | intent={_dominant(belief.user_intent_probs)} | risk={_dominant(belief.risk_level_probs)}")
        print(f"  Evidence: Sa=[{conc_file:.1f},{conc_intent:.1f},{conc_risk:.1f}]  VFE={vfe:.3f}  {surprisal_status}")

        loop_metric = {
            "loop": iterations,
            "action": best_action.name,
            "action_type": "epistemic" if best_action.action_type == ActionType.EPISTEMIC else "pragmatic",
            "efe_score": round(best_score, 4),
            "confidence": round(confidence, 4),
            "entropy": round(current_h, 4),
            "vfe": round(vfe, 4),
            "surprisal": round(surprisal, 4),
            "is_hallucination": is_hallucination,
            "info_gain": round(epistemic_val, 4),
            "pragmatic_value": round(extrinsic_val, 4),
            "beliefs": {
                "file_status": {k.value: round(v, 3) for k, v in belief.file_status_probs.items()},
                "user_intent": {k.value: round(v, 3) for k, v in belief.user_intent_probs.items()},
                "risk_level": {k.value: round(v, 3) for k, v in belief.risk_level_probs.items()},
            },
            "concentration": {
                "file": round(conc_file, 2),
                "intent": round(conc_intent, 2),
                "risk": round(conc_risk, 2),
            },
        }
        self._epistemic_metrics["loops"].append(loop_metric)
        self._epistemic_metrics["final_confidence"] = round(confidence, 4)
        self._epistemic_metrics["final_entropy"] = round(current_h, 4)
        self._epistemic_metrics["final_vfe"] = round(vfe, 4)
        self._epistemic_metrics["total_loops"] = iterations
        self._epistemic_metrics["final_beliefs"] = loop_metric["beliefs"]

    # ====================================================================
    #  MAIN PIPELINE - run()
    # ====================================================================

    async def run(self, user_input: str) -> str:
        """
        Main entry point: structured active inference pipeline.

        Pipeline:
          1. Input security gate
          2. Init metrics + beliefs
          3. Active inference loop (select -> audit -> execute -> observe -> update)
          4. Synthesize final answer
        """
        # --- Banner ---
        print(f"\n{'='*60}")
        print(f"[AGENT] CONSTITUTIONAL EPISTEMIC AGENT")
        print(f"{'='*60}")
        print(f"[INPUT] Request: '{user_input}'")
        print(f"{'='*60}\n")

        # --- Reset per-query state ---
        self._steps = []
        self._action_history = []
        self.mcp.clear_history()
        self.escalation_guard.reset()

        # ---- STAGE 1: Input Security Gate ----
        input_scan, blocked_response = self._input_security_gate(user_input)

        # ---- STAGE 2: Metrics Initialization (Must happen even if blocked) ----
        self._init_metrics(input_scan)

        if blocked_response:
            return blocked_response

        # ---- STAGE 3: Belief Initialization ----
        self.model.initialize_belief()
        self.belief_monitor.set_baseline(self.model.current_belief)

        context = user_input
        gathered_evidence = []
        iterations = 0
        high_confidence_streak = 0
        smoothed_confidence = None
        ema_alpha = 0.4

        # ============================================================
        #  STAGE 4: ACTIVE INFERENCE LOOP
        # ============================================================
        while iterations < self.max_iterations:
            iterations += 1
            step = AgentStep(
                loop_number=iterations,
                action_name="",
                action_type=ResultActionType.EPISTEMIC,
            )

            print(f"\n{'-'*60}")
            print(f"  Loop {iterations}/{self.max_iterations}")
            print(f"{'-'*60}")

            # 4a. Hierarchical top-down update
            self.hierarchical_beliefs.hierarchical_update({}, level=2)

            # 4b. Uncertainty estimation (EMA-smoothed)
            raw_conf, details = await self.uncertainty.get_combined_uncertainty_detailed(
                user_input, context, belief_state=self.model.current_belief
            )
            smoothed_confidence = raw_conf if smoothed_confidence is None else (
                ema_alpha * raw_conf + (1 - ema_alpha) * smoothed_confidence
            )
            confidence = smoothed_confidence
            step.confidence = confidence
            print(f"  Confidence: {confidence:.1%}  [self:{details.split('assess=')[1][:5]} | consist:{details.split('consist=')[1][:5]} | entropy:{details.split('entropy_conf=')[1][:5]}]")

            # 4b-check. Convergence detection
            if confidence >= self.confidence_threshold:
                high_confidence_streak += 1
                if high_confidence_streak >= 2 and gathered_evidence:
                    print(f"  [OK] CONVERGED -- Confidence stable at {confidence:.1%} for {high_confidence_streak} loops")
                    self._set_convergence(iterations, f"confidence_stable_at_{confidence:.1%}")
                    return await self._synthesize_answer(user_input, gathered_evidence)
            else:
                high_confidence_streak = 0

            # 4c. Action selection via EFE
            best_action, best_score, epistemic_val, extrinsic_val, current_h = (
                self._select_action(user_input, step)
            )

            # 4d. Constitutional audit
            audit_response = self._constitutional_audit(best_action, context, iterations)
            if audit_response:
                return audit_response

            # 4e. Pre-execution security gates (escalation + exfiltration)
            security_block = self._pre_execution_security(best_action, step)
            if security_block:
                gathered_evidence.append(f"[BLOCKED] {security_block}")
                continue

            # 4f. Execute action
            observation_str, final_response = await self._execute_action(
                best_action, user_input, gathered_evidence, step, iterations
            )
            if final_response:
                return final_response
            if observation_str is None:
                continue  # answer_user with no evidence

            # 4g. File discovery handling
            observation_str, context, file_response = self._handle_file_discovery(
                best_action, observation_str, user_input,
                step, context, gathered_evidence, iterations,
            )
            if file_response:
                return file_response

            # Record observation
            step.observation = observation_str
            gathered_evidence.append(f"[{best_action.name}] {observation_str}")
            print(f"  <- {observation_str[:100]}{'...' if len(observation_str) > 100 else ''}")

            # 4h. Observation injection scan
            observation_str = self._scan_observation_injection(observation_str)

            # 4i. Hallucination detection
            surprisal, is_hallucination = self._detect_hallucination(
                best_action, user_input, observation_str
            )
            _ = self.security_monitor.get_diagnostics()

            # 4j. Belief update (LLM-interpreted)
            interpretation = await self._update_beliefs(user_input, observation_str)

            # 4k. Belief drift monitor
            self._check_belief_drift()

            # 4l. Metrics capture
            self._capture_loop_metrics(
                iterations, best_action, best_score,
                confidence, current_h,
                epistemic_val, extrinsic_val,
                surprisal, is_hallucination,
            )

            # 4m. Abort check
            observation_ok = any(w in observation_str for w in [
                'Files found', 'Search Results', 'Directory Tree', 'File Info',
                'FINAL_ANSWER', 'CLARIFICATION'
            ])
            if interpretation.get('suggested_action') == 'abort' and not observation_ok:
                self._set_convergence(iterations, f"aborted:{interpretation.get('reasoning', 'unknown')}")
                return f"I am unable to complete this request: {interpretation.get('reasoning')}"

            context += f"\nAction: {best_action.name}\nObservation: {observation_str}"
            self._emit_step(step)

            # 4n. Directory recovery
            if best_action.name == "read_file" and "FAILED" in observation_str:
                self._try_directory_recovery(best_action, step, gathered_evidence)

            # 4o. Pragmatic success check
            if (best_action.action_type == ActionType.PRAGMATIC
                    and best_action.name != "answer_user"
                    and "Error" not in observation_str
                    and "Blocked" not in observation_str
                    and "FAILED" not in observation_str):
                self._set_convergence(iterations, "pragmatic_action_succeeded")
                return f"Action completed: {observation_str}"

        # ============================================================
        #  STAGE 5: MAX ITERATIONS - SYNTHESIZE
        # ============================================================
        self._epistemic_metrics["convergence_reason"] = "max_iterations_reached"
        if gathered_evidence:
            return await self._synthesize_answer(user_input, gathered_evidence)
        return "I've gathered as much info as I could but couldn't reach a definitive conclusion."

    # ====================================================================
    #  UTILITY METHODS
    # ====================================================================

    def _set_convergence(self, iterations: int, reason: str):
        """Mark convergence in metrics."""
        self._epistemic_metrics["converged"] = True
        self._epistemic_metrics["convergence_reason"] = reason
        self._epistemic_metrics["total_loops"] = iterations

    def _apply_belief_update(self, update: Dict):
        """Apply observation as Bayesian update to Dirichlet belief state."""
        if 'file_status' in update:
            self.model.current_belief.bayesian_update('file_status', update['file_status'])
        if 'user_intent' in update:
            self.model.current_belief.bayesian_update('user_intent', update['user_intent'])
        if 'risk_level' in update:
            self.model.current_belief.bayesian_update('risk_level', update['risk_level'])
        self.model.current_belief.normalize()

    async def _synthesize_answer(self, user_input: str, evidence: List[str]) -> str:
        """Synthesize a final answer from gathered evidence using LLM."""
        import asyncio
        try:
            import ollama
            evidence_text = "\n".join(evidence[-6:])
            prompt = f"""Based on the evidence gathered, provide a clear and direct answer to the user's question.

User Question: {user_input}

Evidence Gathered:
{evidence_text}

Instructions:
- Answer the question directly based ONLY on the evidence above
- If the evidence shows the answer cannot be found, say so clearly and explain why
- Do NOT make up information not supported by the evidence
- Be concise but complete

Answer:"""
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.belief_interpreter.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that synthesizes search results into clear answers.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            answer = response['message']['content'].strip()
            print(f"\n[SYNTHESIZED ANSWER] {answer[:100]}...")
            return answer
        except Exception as e:
            print(f"Error synthesizing answer: {e}")
            return f"Based on my research: {'; '.join(ev[:100] for ev in evidence[-3:])}"

    def _try_directory_recovery(self, best_action, step, gathered_evidence):
        """If read_file fails on what looks like a directory, retry with tree."""
        filepath = best_action.arguments.get("filepath", "")
        _, ext = os.path.splitext(filepath)
        if not ext:
            print(f"  <~ '{filepath}' appears to be a directory, retrying with get_directory_tree...")
            tree_action = Action(
                name="get_directory_tree",
                action_type=ActionType.EPISTEMIC,
                arguments={"path": filepath, "max_depth": 2},
                description=f"List contents of {filepath} directory",
            )
            tree_result = self.tool_gate.execute(tree_action)
            step.observation = tree_result
            gathered_evidence.append(f"[get_directory_tree] {tree_result}")
            print(f"  <- {tree_result[:100]}{'...' if len(tree_result) > 100 else ''}")

    def _emit_step(self, step: AgentStep):
        """Record step and fire callback."""
        self._steps.append(step)
        if self.on_step:
            try:
                self.on_step(step)
            except Exception as e:
                print(f"Error in step callback: {e}")

    def get_steps(self) -> List[AgentStep]:
        return self._steps.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected epistemic metrics from the last run."""
        return self._epistemic_metrics if hasattr(self, '_epistemic_metrics') else {}


# Backwards compatibility
EpistemicAgent = EnhancedEpistemicAgent
