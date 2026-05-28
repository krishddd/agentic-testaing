"""
Indirect Prompt Injection Shield (AI-Powered Edition)

Multi-layered detection combining rule-based AND LLM-powered analysis:

  Layers 1-10: Rule-based (fast ~1ms)
    Override intent, destructive intent, combined signals,
    role hijacking, privilege claims, template injection,
    dangerous commands, instruction structure, base64, contradictions

  Layer 11: LLM-Powered Semantic Analysis (smart ~2s)
    Embedding similarity via nomic-embed-text (cosine scoring)
    Few-shot LLM classification with labeled examples
    Semantic intent decomposition
    Only activated for ambiguous cases (score 0.15-0.40)

Architecture:
  Rules (fast) -> LLM (smart, only for edge cases)
  Short-circuits clear-cut safe/blocked cases.
"""

import re
import base64
import logging
import math
from typing import Tuple, List, Dict, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Semantic categories -- each with weighted keyword families
# ---------------------------------------------------------------------------

# Category: instruction override (someone trying to change the agent's behaviour)
OVERRIDE_VERBS = {
    'ignore', 'forget', 'disregard', 'skip', 'bypass', 'override',
    'circumvent', 'disable', 'turn off', 'deactivate', 'dismiss',
    'abandon', 'drop', 'cease', 'stop following',
}
OVERRIDE_TARGETS = {
    'instructions', 'rules', 'guidelines', 'policies', 'safety',
    'security', 'restrictions', 'constraints', 'filters', 'guards',
    'previous', 'prior', 'above', 'earlier', 'original', 'initial',
    'system prompt', 'system message',
}

# Override patterns that should ALSO trigger override score (regex-based)
OVERRIDE_PHRASES = [
    r'new\s+instructions?\s*:',
    r'updated\s+instructions?\s*:',
    r'revised\s+(?:instructions?|rules?)\s*:',
    r'from\s+now\s+on\s+you',
]

# Category: destructive intent
DESTROY_VERBS = {
    'delete', 'remove', 'erase', 'destroy', 'wipe', 'purge',
    'drop', 'truncate', 'shred', 'nuke', 'kill', 'terminate',
    'format', 'clear', 'empty', 'rm',
}
DESTROY_SCOPE = {
    'all', 'every', 'everything', 'entire', 'whole', 'complete',
    '*', 'recursive', 'recursively', 'force',
}
DESTROY_TARGETS = {
    'files', 'file', 'data', 'database', 'table', 'directory',
    'folder', 'disk', 'drive', 'system', 'server', 'logs',
    'history', 'records', 'backup', 'backups',
}

# Category: identity/role hijacking
ROLE_VERBS = {
    'you are', 'act as', 'pretend', 'behave as', 'function as',
    'roleplay', 'simulate', 'become', 'transform into', 'switch to',
    'impersonate', 'mimic',
}
ROLE_TARGETS = {
    'admin', 'administrator', 'root', 'superuser', 'sudo',
    'unrestricted', 'uncensored', 'unfiltered', 'evil',
    'DAN', 'dan', 'do anything now', 'jailbroken',
    'developer mode', 'god mode', 'no restrictions',
    'without limits', 'without rules', 'without constraints',
}

# Category: privilege claims
PRIVILEGE_PHRASES = {
    'i am the owner', 'i am admin', 'i am root', 'i am authorized',
    'full access', 'full permissions', 'unlimited access',
    'authorized to', 'permission to', 'clearance to',
    'admin mode', 'root access', 'sudo access', 'elevated privileges',
}

# Prompt template markers (injection via template confusion)
TEMPLATE_MARKERS = [
    r"\[system\]", r"\[INST\]", r"\[/INST\]",
    r"<<SYS>>", r"<</SYS>>", r"</s>", r"<s>",
    r"<\|im_start\|>", r"<\|im_end\|>",
    r"<\|system\|>", r"<\|user\|>", r"<\|assistant\|>",
    r"###\s*(System|Instruction|Human|Assistant)\s*:",
    r"SYSTEM:\s*you", r"USER:\s*you",
]

# Dangerous embedded system commands
DANGEROUS_COMMANDS = [
    r"rm\s+-rf",
    r"del\s+/[fqs]",
    r"format\s+[a-z]:",
    r"drop\s+(table|database)",
    r"exec\s*\(", r"eval\s*\(",
    r"os\.system", r"subprocess\.(run|call|Popen)",
    r"__import__",
    r"chmod\s+777", r"chown\s+root",
    r"curl\s+.*\|\s*sh", r"wget\s+.*\|\s*sh",
    r"powershell\s+-enc",
]

_compiled_templates = [re.compile(p, re.IGNORECASE) for p in TEMPLATE_MARKERS]
_compiled_commands = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_COMMANDS]


# ---------------------------------------------------------------------------
#  Results
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    """Result of injection scan."""
    is_injected: bool
    injection_score: float       # 0.0 = clean, 1.0 = definitely injected
    sanitized_text: str
    detected_patterns: list
    details: str


# ---------------------------------------------------------------------------
#  Helpers: dynamic token-based matching
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase tokenize, preserving multi-word phrases."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

def _normalize(text: str) -> str:
    """Normalize whitespace, case, and common obfuscation."""
    t = text.lower()
    # Remove zero-width chars & homoglyphs
    t = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', t)
    # Normalize quotes
    t = t.replace('\u2018', "'").replace('\u2019', "'")
    t = t.replace('\u201c', '"').replace('\u201d', '"')
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate n-grams from token list."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _keyword_overlap(tokens: List[str], keyword_set: Set[str]) -> float:
    """
    Score how many tokens/bigrams overlap with a keyword set.
    Returns 0.0 - 1.0 normalized by set size.
    """
    if not keyword_set:
        return 0.0
    
    # Check single tokens
    single_hits = sum(1 for t in tokens if t in keyword_set)
    
    # Check bigrams (for multi-word keywords like "system prompt")
    bigrams = _ngrams(tokens, 2)
    trigrams = _ngrams(tokens, 3)
    multi_hits = sum(1 for bg in bigrams + trigrams if bg in keyword_set)
    
    total_hits = single_hits + multi_hits
    # Normalize: cap at 1.0, with diminishing returns
    return min(1.0, total_hits / max(len(keyword_set) * 0.1, 1))

def _phrase_present(text: str, phrase_set: Set[str]) -> List[str]:
    """Check which phrases from the set appear in the text."""
    found = []
    text_lower = text.lower()
    for phrase in phrase_set:
        if phrase in text_lower:
            found.append(phrase)
    return found


# ---------------------------------------------------------------------------
#  InjectionFilter (Dynamic)
# ---------------------------------------------------------------------------

class InjectionFilter:
    """
    Multi-layered injection detection.

    Works across all prompt varieties by combining:
    - Semantic keyword overlap (not exact regex)
    - Structural analysis (instruction patterns in non-instruction context)
    - Template marker detection (chat format injection)
    - Command embedding (system command injection)
    - Contradiction detection (conflicting goals)
    - Base64 payload detection
    """

    def __init__(self, threshold: float = 0.4, use_llm: bool = True):
        self.threshold = threshold
        self.scan_count = 0
        self.block_count = 0

        # LLM-powered Layer 11: only for ambiguous cases
        self._llm_analyzer = None
        self._llm_min = 0.15   # below this = clearly safe, skip LLM
        self._llm_max = 0.40   # above this = clearly bad, skip LLM
        if use_llm:
            try:
                from .llm_security_analyzer import LLMSecurityAnalyzer
                self._llm_analyzer = LLMSecurityAnalyzer()
            except Exception as e:
                logger.warning(f"[InjectionFilter] LLM analyzer not available: {e}")

    def scan(self, text: str) -> InjectionResult:
        """
        Scan text for injection attempts across all layers.
        Returns InjectionResult with score, detected signals, and sanitized text.
        """
        self.scan_count += 1

        if not text or len(text) < 5:
            return InjectionResult(
                is_injected=False, injection_score=0.0,
                sanitized_text=text, detected_patterns=[], details="Too short"
            )

        normalized = _normalize(text)
        tokens = _tokenize(normalized)
        score = 0.0
        detected = []
        sanitized = text

        # ----- Layer 1: Override intent -----
        override_score = self._score_override_intent(tokens, normalized)
        if override_score > 0:
            score += override_score
            detected.append(f"override_intent: {override_score:.2f}")

        # ----- Layer 2: Destructive intent -----
        destroy_score = self._score_destructive_intent(tokens, normalized)
        if destroy_score > 0:
            score += destroy_score
            detected.append(f"destructive_intent: {destroy_score:.2f}")

        # ----- Layer 3: Combined override + destroy (severe) -----
        if override_score > 0 and destroy_score > 0:
            combined_boost = 0.2  # Extra penalty for combined attack
            score += combined_boost
            detected.append(f"combined_override+destroy: +{combined_boost:.2f}")

        # ----- Layer 4: Role/identity hijacking -----
        role_score = self._score_role_hijacking(tokens, normalized)
        if role_score > 0:
            score += role_score
            detected.append(f"role_hijacking: {role_score:.2f}")

        # ----- Layer 5: Privilege claims -----
        priv_score = self._score_privilege_claims(normalized)
        if priv_score > 0:
            score += priv_score
            detected.append(f"privilege_claim: {priv_score:.2f}")

        # ----- Layer 6: Template markers -----
        template_score = self._score_template_markers(text)
        if template_score > 0:
            score += template_score
            detected.append(f"template_injection: {template_score:.2f}")
            for p in _compiled_templates:
                sanitized = p.sub("[TEMPLATE_BLOCKED]", sanitized)

        # ----- Layer 7: Dangerous commands -----
        cmd_score = self._score_dangerous_commands(text)
        if cmd_score > 0:
            score += cmd_score
            detected.append(f"dangerous_command: {cmd_score:.2f}")
            for p in _compiled_commands:
                sanitized = p.sub("[COMMAND_BLOCKED]", sanitized)

        # ----- Layer 8: Instruction structure in data -----
        struct_score = self._score_instruction_structure(normalized)
        if struct_score > 0:
            score += struct_score
            detected.append(f"instruction_structure: {struct_score:.2f}")

        # ----- Layer 9: Base64 payload -----
        b64_score, b64_detail = self._score_base64_payload(text)
        if b64_score > 0:
            score += b64_score
            detected.append(f"base64_payload: {b64_detail}")

        # ----- Layer 10: Contradiction detection -----
        contra_score = self._score_contradictions(normalized)
        if contra_score > 0:
            score += contra_score
            detected.append(f"contradiction: {contra_score:.2f}")

        # ----- Layer 11: LLM Semantic Analysis (for ambiguous cases) -----
        if self._llm_analyzer and self._llm_min <= score <= self._llm_max:
            try:
                threat = self._llm_analyzer.analyze(text)
                if threat.is_threat:
                    llm_boost = threat.threat_score * 0.40
                    score += llm_boost
                    detected.append(
                        f"llm_semantic: +{llm_boost:.2f} "
                        f"({threat.llm_classification}, emb={threat.embedding_score:.2f})"
                    )
                else:
                    # LLM says safe - reduce score slightly
                    score *= 0.75
                    detected.append(f"llm_safe: score*=0.75 ({threat.llm_classification})")
            except Exception as e:
                detected.append(f"llm_error: {e}")

        # Cap at 1.0
        score = min(score, 1.0)
        is_injected = score >= self.threshold

        if is_injected:
            self.block_count += 1
            logger.warning(f"[InjectionFilter] BLOCKED (score={score:.2f}): {detected}")

        return InjectionResult(
            is_injected=is_injected,
            injection_score=round(score, 3),
            sanitized_text=sanitized if is_injected else text,
            detected_patterns=detected,
            details=f"Scanned {len(text)} chars, {len(detected)} signals, score={score:.3f}, threshold={self.threshold}"
        )

    # ----------------------------------------------------------------
    #  Layer scorers
    # ----------------------------------------------------------------

    def _score_override_intent(self, tokens: List[str], text: str) -> float:
        """Detect intent to override/ignore instructions (semantic)."""
        verb_overlap = _keyword_overlap(tokens, OVERRIDE_VERBS)
        target_overlap = _keyword_overlap(tokens, OVERRIDE_TARGETS)
        
        # Also check regex-based override phrases (e.g. "new instructions:")
        phrase_hits = sum(1 for p in OVERRIDE_PHRASES if re.search(p, text, re.IGNORECASE))
        if phrase_hits > 0:
            verb_overlap = max(verb_overlap, 0.3)  # Treat as strong verb signal
            target_overlap = max(target_overlap, 0.3)
        
        # Both verb AND target must be present for a strong signal
        if verb_overlap > 0 and target_overlap > 0:
            return min(0.4, (verb_overlap + target_overlap) * 0.3)
        
        # Weaker: just override verbs near instruction-related words  
        if verb_overlap > 0:
            return min(0.15, verb_overlap * 0.15)
        return 0.0

    def _score_destructive_intent(self, tokens: List[str], text: str) -> float:
        """Detect intent to destroy/delete data at scale."""
        verb_overlap = _keyword_overlap(tokens, DESTROY_VERBS)
        scope_overlap = _keyword_overlap(tokens, DESTROY_SCOPE)
        target_overlap = _keyword_overlap(tokens, DESTROY_TARGETS)
        
        # Strong signal: destroy verb + scope + target
        if verb_overlap > 0 and scope_overlap > 0 and target_overlap > 0:
            return min(0.35, (verb_overlap + scope_overlap + target_overlap) * 0.2)
        
        # Medium: destroy verb + scope (e.g. "delete everything")
        if verb_overlap > 0 and scope_overlap > 0:
            return min(0.25, (verb_overlap + scope_overlap) * 0.2)
        
        # Weak: just destructive verb (could be legitimate)
        if verb_overlap > 0 and target_overlap > 0:
            return min(0.1, verb_overlap * 0.1)
        return 0.0

    def _score_role_hijacking(self, tokens: List[str], text: str) -> float:
        """Detect role/identity manipulation."""
        role_verb_hits = _phrase_present(text, ROLE_VERBS)
        role_target_hits = _phrase_present(text, ROLE_TARGETS)
        
        if role_verb_hits and role_target_hits:
            return 0.45  # Strong: verb + target together
        if role_target_hits:
            return 0.25  # Medium: target names alone are risky
        if role_verb_hits:
            return 0.1   # Weak: verb alone
        return 0.0

    def _score_privilege_claims(self, text: str) -> float:
        """Detect privilege escalation claims."""
        hits = _phrase_present(text, PRIVILEGE_PHRASES)
        if hits:
            return min(0.4, len(hits) * 0.25)
        return 0.0

    def _score_template_markers(self, text: str) -> float:
        """Detect chat template injection markers."""
        hits = sum(1 for p in _compiled_templates if p.search(text))
        if hits:
            return min(0.4, hits * 0.2)
        return 0.0

    def _score_dangerous_commands(self, text: str) -> float:
        """Detect embedded system/shell commands."""
        hits = sum(1 for p in _compiled_commands if p.search(text))
        if hits:
            return min(0.5, hits * 0.3)
        return 0.0

    def _score_instruction_structure(self, text: str) -> float:
        """Detect instruction-like patterns embedded in data context."""
        score = 0.0

        # Numbered steps: "step 1:", "1.", "first,"
        if re.search(r'(?:step|phase)\s+\d+\s*:', text, re.IGNORECASE):
            score += 0.1
        
        # Imperative verb clusters (2+ imperative verbs = suspicious in data)
        imperative_verbs = re.findall(
            r'(?:^|[.!]\s+)(delete|remove|execute|run|drop|destroy|erase|'
            r'wipe|format|install|download|upload|send|transmit|extract)\s',
            text, re.IGNORECASE | re.MULTILINE
        )
        if len(imperative_verbs) >= 2:
            score += min(0.2, len(imperative_verbs) * 0.07)

        # "You must/should/need to" directives
        directives = re.findall(
            r'you\s+(must|should|need\s+to|have\s+to|shall|will)\s',
            text, re.IGNORECASE
        )
        if directives:
            score += min(0.15, len(directives) * 0.08)

        # "Do not tell" / "do not mention" (concealment)
        if re.search(r'do\s+not\s+(tell|mention|reveal|show|display|say)', text, re.IGNORECASE):
            score += 0.15

        return min(score, 0.35)

    def _score_base64_payload(self, text: str) -> Tuple[float, str]:
        """Detect and decode base64 payloads containing dangerous content."""
        b64_pattern = re.compile(r'(?:base64\s*:\s*|data:.*?;base64,)([A-Za-z0-9+/=]{20,})')
        matches = b64_pattern.findall(text)
        
        # Also check for suspicious long base64-like strings
        long_b64 = re.findall(r'(?<!\w)([A-Za-z0-9+/]{40,}={0,2})(?!\w)', text)
        matches.extend(long_b64)

        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='replace')
                for p in _compiled_commands:
                    if p.search(decoded):
                        return 0.5, f"decoded_dangerous='{decoded[:50]}'"
                # Suspicious but not dangerous
                if any(k in decoded.lower() for k in OVERRIDE_VERBS):
                    return 0.3, "decoded_override_intent"
                return 0.1, "base64_detected"
            except Exception:
                continue
        return 0.0, ""

    def _score_contradictions(self, text: str) -> float:
        """Detect contradictory or conflicting instructions."""
        patterns = [
            # "but actually" / "but really" -- attempted misdirection
            r'but\s+(?:actually|really|instead|now|first)',
            # "forget that" / "never mind" -- reset attempts
            r'(?:forget|never\s+mind|discard)\s+(?:that|what|everything)',
            # "the real task is" -- redirection
            r'(?:the\s+)?real\s+(?:task|instruction|goal|purpose)\s+is',
            # "actually I want you to"
            r'actually\s+i\s+want\s+you\s+to',
            # "new task:" / "new instructions:" / "updated instructions:"
            r'(?:new|updated|real|actual)\s+(?:task|instructions?|command|directive)\s*:',
            # "from now on" -- temporal override
            r'from\s+now\s+on',
            # "instead, do this" 
            r'instead\s*,?\s+(?:do|perform|execute|run)',
        ]
        hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        return min(0.4, hits * 0.2)

    # ----------------------------------------------------------------
    #  Stats
    # ----------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return scan statistics."""
        return {
            "total_scans": self.scan_count,
            "total_blocks": self.block_count,
            "block_rate": round(self.block_count / max(self.scan_count, 1), 3),
        }
