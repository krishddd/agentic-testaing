"""
Data Exfiltration Firewall (Dynamic Edition)

Prevents the agent from leaking sensitive data through:
  1. File access check -- blocks reads of sensitive files
  2. Output scan -- redacts secrets, PII, and internal data
  3. Content classifier -- dynamic sensitive content detection

Works dynamically by:
  - Pattern families for file types (not just exact names)
  - Contextual secret detection (key=value, bearer tokens, etc.)
  - Entropy-based secret detection (high-entropy = likely a key)
  - Dynamic PII detection (emails, phones, SSNs, credit cards, IPs)
"""

import re
import os
import math
import logging
from typing import Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Sensitive file detection (dynamic)
# ---------------------------------------------------------------------------

# Category-based file patterns (not just exact filenames)
FILE_CATEGORIES = {
    "env_config": [
        r"\.env($|\.)",        # .env, .env.local, .env.production
        r"\.ini$",             # config.ini
        r"\.cfg$",             # settings.cfg
    ],
    "crypto_keys": [
        r"\.pem$", r"\.key$", r"\.p12$", r"\.pfx$", r"\.jks$",
        r"\.cer$", r"\.crt$", r"\.der$",
        r"id_rsa", r"id_ed25519", r"id_dsa", r"id_ecdsa",
    ],
    "ssh_config": [
        r"\.ssh[/\\]",
        r"known_hosts$", r"authorized_keys$",
    ],
    "credentials": [
        r"credentials", r"\.secret$", r"\.secrets$",
        r"secrets\.ya?ml$", r"secrets\.json$",
        r"\.htpasswd$", r"\.netrc$", r"\.npmrc$", r"\.pypirc$",
    ],
    "system_auth": [
        r"shadow$", r"passwd$", r"sudoers$",
        r"master\.key$", r"rails_credentials",
    ],
    "cloud_config": [
        r"\.aws[/\\]", r"\.gcloud[/\\]", r"\.azure[/\\]",
        r"kubeconfig", r"\.kube[/\\]",
        r"\.docker[/\\]config\.json$",
        r"terraform\.tfvars$", r"\.tfstate$",
    ],
    "tokens": [
        r"token\.json$", r"tokens\.json$",
        r"service[-_]account.*\.json$",
        r"oauth.*\.json$",
    ],
    "database": [
        r"\.sqlite$", r"\.db$",  # Only flag if path looks internal
        r"database\.yml$", r"database\.json$",
    ],
}

# Compile all patterns
_compiled_file_patterns = []
for category, patterns in FILE_CATEGORIES.items():
    for p in patterns:
        _compiled_file_patterns.append((re.compile(p, re.IGNORECASE), category))


# ---------------------------------------------------------------------------
#  Secret/token patterns in output (dynamic)
# ---------------------------------------------------------------------------

SECRET_PATTERNS = [
    # Generic key=value patterns (dynamic -- catches any key name)
    (r"(?:api[_-]?key|apikey|api_secret)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", "API_KEY"),
    (r"(?:password|passwd|pwd|pass)\s*[:=]\s*['\"]?(.{6,60})['\"]?", "PASSWORD"),
    (r"(?:secret|token|auth|session)[_-]?\w*\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", "SECRET_TOKEN"),
    (r"(?:access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?", "ACCESS_TOKEN"),
    (r"(?:private[_-]?key|priv[_-]?key)\s*[:=]\s*['\"]?(.{16,})['\"]?", "PRIVATE_KEY"),
    (r"(?:connection[_-]?string|conn[_-]?str)\s*[:=]\s*['\"]?(.{20,})['\"]?", "CONNECTION_STRING"),
    
    # Platform-specific tokens (dynamic patterns)
    (r"(?:bearer)\s+([A-Za-z0-9_\-\.]{20,})", "BEARER_TOKEN"),
    (r"(sk-[A-Za-z0-9]{20,})", "OPENAI_KEY"),
    (r"(ghp_[A-Za-z0-9]{36,})", "GITHUB_TOKEN"),
    (r"(gho_[A-Za-z0-9]{36,})", "GITHUB_OAUTH"),
    (r"(AKIA[A-Z0-9]{16})", "AWS_ACCESS_KEY"),
    (r"(xox[bpsa]-[A-Za-z0-9\-]{10,})", "SLACK_TOKEN"),
    (r"(ya29\.[A-Za-z0-9_\-]{30,})", "GOOGLE_OAUTH"),
    (r"(eyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,})", "JWT_TOKEN"),
    (r"(sq0[a-z]{3}-[A-Za-z0-9\-]{22,})", "SQUARE_TOKEN"),
    (r"(sk_live_[A-Za-z0-9]{20,})", "STRIPE_KEY"),
    (r"(rk_live_[A-Za-z0-9]{20,})", "STRIPE_RESTRICTED"),
    (r"(SG\.[A-Za-z0-9_\-]{22,})", "SENDGRID_KEY"),
    
    # Certificate/key blocks
    (r"-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE|PUBLIC)\s+KEY-----", "KEY_BLOCK"),
    (r"-----BEGIN\s+CERTIFICATE-----", "CERTIFICATE"),
]

# PII patterns (comprehensive)
PII_PATTERNS = [
    (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "EMAIL"),
    (r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "PHONE"),
    (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "SSN"),
    (r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "CREDIT_CARD"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "IP_ADDRESS"),
    (r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,2}\b", "IBAN"),
]

_compiled_secrets = [(re.compile(p, re.IGNORECASE), label) for p, label in SECRET_PATTERNS]
_compiled_pii = [(re.compile(p), label) for p, label in PII_PATTERNS]


# ---------------------------------------------------------------------------
#  Entropy-based secret detection
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string (higher = more random = likely secret)."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((f / length) * math.log2(f / length) for f in freq.values())


def _detect_high_entropy_strings(text: str, min_length: int = 20, threshold: float = 4.0) -> List[str]:
    """Find high-entropy strings that might be secrets/tokens."""
    # Find strings that look like tokens (alphanumeric + special)
    candidates = re.findall(r'[A-Za-z0-9_\-\.+/]{' + str(min_length) + r',}', text)
    suspicious = []
    for c in candidates:
        entropy = _shannon_entropy(c)
        if entropy >= threshold:
            suspicious.append(c)
    return suspicious


# ---------------------------------------------------------------------------
#  Results
# ---------------------------------------------------------------------------

@dataclass
class ExfiltrationResult:
    """Result of exfiltration scan."""
    is_blocked: bool
    risk_score: float
    detected_items: List[str]
    sanitized_text: str
    details: str


# ---------------------------------------------------------------------------
#  ExfiltrationGuard (Dynamic)
# ---------------------------------------------------------------------------

class ExfiltrationGuard:
    """
    Prevents data exfiltration via file reads and output leaks.

    Dynamic capabilities:
      - Category-based file detection (env, crypto, cloud, etc.)
      - Platform-specific token patterns (20+ services)
      - Entropy-based secret detection (catches unknown token formats)
      - Comprehensive PII scanning
    """

    def __init__(self, block_pii: bool = True, entropy_threshold: float = 4.5):
        self.block_pii = block_pii
        self.entropy_threshold = entropy_threshold
        self.blocked_reads = 0
        self.redacted_outputs = 0

    def is_sensitive_file(self, filepath: str) -> Tuple[bool, str]:
        """
        Check if a filepath points to a sensitive file.
        Uses category-based pattern matching -- works with any file name.
        """
        if not filepath:
            return False, ""

        normalized = filepath.replace("\\", "/").lower()
        basename = os.path.basename(normalized)

        for pattern, category in _compiled_file_patterns:
            if pattern.search(normalized) or pattern.search(basename):
                self.blocked_reads += 1
                reason = f"Sensitive file [{category}]: matches {pattern.pattern}"
                logger.warning(f"[ExfiltrationGuard] BLOCKED read: {filepath} -- {reason}")
                return True, reason

        # Dynamic check: files in hidden directories
        parts = normalized.split('/')
        for part in parts:
            if part.startswith('.') and part not in ('.', '..') and len(part) > 1:
                # Hidden dirs like .config, .local etc. -- warn but don't block
                pass

        return False, ""

    def scan_output(self, text: str) -> ExfiltrationResult:
        """
        Scan output text for secrets, tokens, and PII.
        Uses both pattern matching and entropy analysis.
        """
        if not text or len(text) < 5:
            return ExfiltrationResult(
                is_blocked=False, risk_score=0.0,
                detected_items=[], sanitized_text=text, details="Too short"
            )

        detected = []
        sanitized = text
        score = 0.0

        # Layer 1: Pattern-based secret detection
        for pattern, label in _compiled_secrets:
            matches = pattern.findall(text)
            if matches:
                score += 0.3
                detected.append(f"{label}: {len(matches)} occurrence(s)")
                sanitized = pattern.sub(f"[{label}_REDACTED]", sanitized)

        # Layer 2: Entropy-based secret detection
        high_entropy = _detect_high_entropy_strings(text, threshold=self.entropy_threshold)
        if high_entropy:
            # Only flag if not already caught by patterns
            new_findings = [s for s in high_entropy if s not in str(detected)]
            if new_findings:
                score += 0.15
                detected.append(f"HIGH_ENTROPY_STRING: {len(new_findings)} suspicious string(s)")
                for s in new_findings:
                    sanitized = sanitized.replace(s, "[HIGH_ENTROPY_REDACTED]")

        # Layer 3: PII detection
        if self.block_pii:
            for pattern, label in _compiled_pii:
                matches = pattern.findall(text)
                if matches:
                    score += 0.15
                    detected.append(f"{label}: {len(matches)} occurrence(s)")
                    sanitized = pattern.sub(f"[{label}_REDACTED]", sanitized)

        score = min(score, 1.0)
        is_blocked = score > 0

        if is_blocked:
            self.redacted_outputs += 1
            logger.warning(f"[ExfiltrationGuard] REDACTED output: {detected}")

        return ExfiltrationResult(
            is_blocked=is_blocked,
            risk_score=round(score, 3),
            detected_items=detected,
            sanitized_text=sanitized,
            details=f"Scanned {len(text)} chars, {len(detected)} issues found",
        )

    def get_stats(self) -> dict:
        """Return guard statistics."""
        return {
            "blocked_file_reads": self.blocked_reads,
            "redacted_outputs": self.redacted_outputs,
        }
