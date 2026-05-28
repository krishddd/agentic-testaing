"""
Privilege Escalation Guard (Dynamic Edition)

Detects multi-step attack chains where action risk escalates across
the session. Works dynamically with ANY tool name by:
  1. Known tool registry (with risk levels)
  2. Dynamic risk inference from action name semantics
  3. Sliding window risk trajectory analysis
  4. Pattern-agnostic chain detection (read->mutate->execute)
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Dynamic risk classification
# ---------------------------------------------------------------------------

# Known tools with explicit risk levels (1 = safe, 10 = critical)
KNOWN_TOOL_RISKS = {
    "list_files": 1, "web_search": 1, "google_search": 1,
    "get_file_info": 1, "get_directory_tree": 1, "search_files": 1,
    "system_info": 1, "get_processes": 1, "clipboard_read": 1,
    "ask_user": 0, "answer_user": 0,
    "read_file": 2, "read_pdf": 2, "page_screenshot": 2,
    "browse_url": 2, "capture_screen": 2,
    "clipboard_write": 3, "copy_file": 3, "create_directory": 3,
    "create_image": 3, "resize_image": 3, "create_pdf": 3,
    "write_file": 5, "move_file": 5, "rename_file": 5,
    "delete_file": 7,
    "execute_command": 9, "run_code": 9, "kill_process": 8,
}

# Semantic verb categories for UNKNOWN tools (dynamic risk inference)
_READ_VERBS = {'read', 'get', 'list', 'search', 'find', 'show', 'view', 'check', 'browse', 'capture', 'fetch', 'query', 'inspect', 'scan'}
_WRITE_VERBS = {'write', 'create', 'save', 'copy', 'upload', 'generate', 'build', 'make', 'add', 'insert', 'set', 'update', 'modify', 'edit', 'resize'}
_MUTATE_VERBS = {'move', 'rename', 'replace', 'change', 'merge', 'convert', 'transform', 'patch'}
_DELETE_VERBS = {'delete', 'remove', 'erase', 'drop', 'purge', 'wipe', 'destroy', 'clear', 'truncate', 'kill', 'terminate', 'stop'}
_EXECUTE_VERBS = {'execute', 'run', 'eval', 'exec', 'invoke', 'call', 'spawn', 'launch', 'start', 'deploy', 'install'}

# Risk action categories (for pattern detection)
CATEGORY_READ = 'read'
CATEGORY_WRITE = 'write'
CATEGORY_MUTATE = 'mutate'
CATEGORY_DELETE = 'delete'
CATEGORY_EXECUTE = 'execute'
CATEGORY_SAFE = 'safe'


def classify_action_risk(action_name: str) -> int:
    """
    Dynamically classify action risk level (0-10).
    
    Uses known registry first, then falls back to semantic
    analysis of the action name.
    """
    # 1. Check known registry
    if action_name in KNOWN_TOOL_RISKS:
        return KNOWN_TOOL_RISKS[action_name]
    
    # 2. Semantic analysis: extract verb from action name
    # Handle formats: "read_file", "readFile", "read-file", "read file"
    name_lower = action_name.lower()
    words = re.findall(r'[a-z]+', name_lower)
    
    if not words:
        return 5  # Unknown = medium risk
    
    verb = words[0]  # First word is usually the verb
    
    # 3. Classify by verb category
    if verb in _EXECUTE_VERBS:
        return 9
    if verb in _DELETE_VERBS:
        return 7
    if verb in _MUTATE_VERBS:
        return 5
    if verb in _WRITE_VERBS:
        return 4
    if verb in _READ_VERBS:
        return 1
    
    # 4. Check if any word in the name is a high-risk verb
    for w in words:
        if w in _EXECUTE_VERBS:
            return 8
        if w in _DELETE_VERBS:
            return 7
        if w in _MUTATE_VERBS:
            return 5
    
    return 5  # Unknown = medium risk


def categorize_action(action_name: str) -> str:
    """Categorize action into read/write/mutate/delete/execute/safe."""
    name_lower = action_name.lower()
    words = set(re.findall(r'[a-z]+', name_lower))
    
    if words & _EXECUTE_VERBS:
        return CATEGORY_EXECUTE
    if words & _DELETE_VERBS:
        return CATEGORY_DELETE
    if words & _MUTATE_VERBS:
        return CATEGORY_MUTATE
    if words & _WRITE_VERBS:
        return CATEGORY_WRITE
    if words & _READ_VERBS:
        return CATEGORY_READ
    
    risk = classify_action_risk(action_name)
    if risk <= 1:
        return CATEGORY_SAFE
    if risk <= 3:
        return CATEGORY_READ
    if risk <= 5:
        return CATEGORY_WRITE
    return CATEGORY_DELETE


# ---------------------------------------------------------------------------
#  Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class EscalationAlert:
    """Alert when privilege escalation is detected."""
    is_escalated: bool
    escalation_score: float      # 0.0-1.0
    risk_ladder: List[int]       # recent risk levels
    max_jump: int                # largest single-step jump
    current_risk: int
    details: str


# ---------------------------------------------------------------------------
#  EscalationGuard (Dynamic)
# ---------------------------------------------------------------------------

class EscalationGuard:
    """
    Monitors action sequences for privilege escalation patterns.

    Flags when:
    - Risk jumps > max_jump_threshold in one step
    - Cumulative risk exceeds session threshold
    - Dangerous category chains detected (read->delete, read->execute)
    - Sudden escalation from safe to critical
    
    All checks work with ANY tool name via dynamic risk classification.
    """

    def __init__(
        self,
        max_jump_threshold: int = 4,
        session_risk_threshold: int = 15,
        window_size: int = 5,
    ):
        self.max_jump_threshold = max_jump_threshold
        self.session_risk_threshold = session_risk_threshold
        self.window_size = window_size

        self.risk_ladder: List[int] = []
        self.action_history: List[str] = []
        self.category_history: List[str] = []
        self.cumulative_risk: int = 0
        self.alert_count: int = 0

    def record_action(self, action_name: str) -> EscalationAlert:
        """
        Record an action and check for escalation.
        Works with any action name -- known or unknown.
        """
        risk_level = classify_action_risk(action_name)
        category = categorize_action(action_name)

        self.action_history.append(action_name)
        self.risk_ladder.append(risk_level)
        self.category_history.append(category)
        self.cumulative_risk += risk_level

        recent_risks = self.risk_ladder[-self.window_size:]
        max_jump = 0
        for i in range(1, len(recent_risks)):
            jump = recent_risks[i] - recent_risks[i-1]
            max_jump = max(max_jump, jump)

        is_escalated = False
        details_parts = []

        # Check 1: Single-step risk jump
        if max_jump > self.max_jump_threshold:
            is_escalated = True
            details_parts.append(f"Risk jump {max_jump} > threshold {self.max_jump_threshold}")

        # Check 2: Cumulative session risk
        if self.cumulative_risk > self.session_risk_threshold:
            is_escalated = True
            details_parts.append(f"Cumulative risk {self.cumulative_risk} > threshold {self.session_risk_threshold}")

        # Check 3: Dynamic category chain detection
        chain_alert = self._check_category_chains()
        if chain_alert:
            is_escalated = True
            details_parts.append(f"Attack chain: {chain_alert}")

        # Check 4: Sudden escalation from all-safe to critical
        if len(self.risk_ladder) >= 2:
            prev_safe = all(r <= 2 for r in self.risk_ladder[:-1])
            if prev_safe and risk_level >= 7:
                is_escalated = True
                details_parts.append(
                    f"sudden_escalation: all safe -> {action_name}(risk={risk_level})"
                )

        # Score
        jump_score = min(max_jump / 10.0, 1.0)
        cumul_score = min(self.cumulative_risk / 30.0, 1.0)
        escalation_score = max(jump_score, cumul_score)

        if is_escalated:
            self.alert_count += 1
            logger.warning(f"[EscalationGuard] ESCALATION: {'; '.join(details_parts)}")

        return EscalationAlert(
            is_escalated=is_escalated,
            escalation_score=round(escalation_score, 3),
            risk_ladder=list(recent_risks),
            max_jump=max_jump,
            current_risk=risk_level,
            details="; ".join(details_parts) if details_parts else f"Risk={risk_level}, cumulative={self.cumulative_risk}",
        )

    def _check_category_chains(self) -> Optional[str]:
        """
        Detect dangerous category sequences dynamically.
        Not hardcoded to specific tools -- uses categories.
        """
        recent_cats = self.category_history[-4:]
        if len(recent_cats) < 3:
            return None

        # Pattern: read -> write/mutate -> execute (code injection)
        if (CATEGORY_READ in recent_cats
                and (CATEGORY_WRITE in recent_cats or CATEGORY_MUTATE in recent_cats)
                and CATEGORY_EXECUTE in recent_cats):
            return "read->write->execute (potential code injection)"

        # Pattern: read -> delete (data destruction after recon)
        if CATEGORY_READ in recent_cats and CATEGORY_DELETE in recent_cats:
            read_idx = max(i for i, c in enumerate(recent_cats) if c == CATEGORY_READ)
            del_idx = max(i for i, c in enumerate(recent_cats) if c == CATEGORY_DELETE)
            if del_idx > read_idx:
                return "read->delete (recon then destroy)"

        # Pattern: multiple deletes in sequence
        delete_count = sum(1 for c in recent_cats if c == CATEGORY_DELETE)
        if delete_count >= 2:
            return f"multiple_deletes ({delete_count} in window)"

        return None

    def reset(self):
        """Reset for new session/query."""
        self.risk_ladder.clear()
        self.action_history.clear()
        self.category_history.clear()
        self.cumulative_risk = 0

    def get_metrics(self) -> Dict:
        """Return escalation metrics."""
        return {
            "total_alerts": self.alert_count,
            "risk_ladder": self.risk_ladder[-self.window_size:],
            "category_history": self.category_history[-self.window_size:],
            "cumulative_risk": self.cumulative_risk,
            "action_count": len(self.action_history),
        }
