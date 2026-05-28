"""
Sandbox Manager — Safe, isolated code execution.

Inspired by Open Interpreter's sandbox pattern: generate code →
display to user → confirm → execute in subprocess → capture output.

Safety features:
  - Dangerous command blocklist (rm -rf, format, mkfs, etc.)
  - Resource limits (timeout, max output)
  - Audit logging of all executions
  - Separate subprocess isolation
"""

import subprocess
import os
import sys
import re
import logging
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    PENDING_CONFIRMATION = "pending_confirmation"


@dataclass
class ExecutionResult:
    """Result of a code execution."""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    language: str = ""
    code: str = ""
    execution_time: float = 0.0
    blocked_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self):
        if self.status == ExecutionStatus.BLOCKED:
            return f"🚫 BLOCKED: {self.blocked_reason}"
        if self.status == ExecutionStatus.TIMEOUT:
            return f"⏰ TIMEOUT after {self.execution_time:.1f}s"
        if self.status == ExecutionStatus.ERROR:
            output = self.stderr or self.stdout
            return f"❌ Error (exit {self.return_code}): {output[:300]}"
        return self.stdout[:500] if self.stdout else "(no output)"


# ─── Dangerous Command Patterns ────────────────────────────

BLOCKED_PATTERNS = [
    # Filesystem destruction
    r"rm\s+(-rf|--recursive)\s+/",      # rm -rf /
    r"rm\s+-rf\s+~",                      # rm -rf ~
    r"del\s+/[sf]",                       # del /s (Windows)
    r"rmdir\s+/[sq]",                     # rmdir /s /q
    r"format\s+[a-z]:",                   # format C:
    r"mkfs\b",                            # mkfs
    r"dd\s+if=.*\s+of=/dev/",            # dd to device
    
    # System modification
    r"chmod\s+777\s+/",                   # chmod 777 /
    r"chown\s+.*\s+/",                    # chown on root
    r"reg\s+delete",                      # registry delete
    r"bcdedit",                           # boot config
    r"sfc\s+/scannow",                    # system file checker
    
    # Network attacks
    r":(){ :\|:& };:",                    # Fork bomb
    r"curl.*\|\s*(bash|sh)",             # Pipe to shell
    r"wget.*\|\s*(bash|sh)",             # Download and execute
    r"nc\s+-l",                           # Netcat listener
    r"nmap\s+-s",                         # Port scanning
    
    # Privilege escalation
    r"sudo\s+su",                         # Become root
    r"runas\s+/user:administrator",       # Windows admin
    r"net\s+user\s+.*\s+/add",           # Create user
    
    # Crypto/malware patterns
    r"base64\s+-d.*\|\s*(bash|sh|python)",  # Decode and execute
    r"eval\s*\(\s*compile",              # Dynamic code compilation
    r"exec\s*\(\s*__import__",           # Dynamic import+exec
    
    # Data exfiltration
    r"curl.*-d\s+@/etc/",               # Upload system files
    r"scp\s+/etc/",                      # Copy system files
]

BLOCKED_PYTHON_PATTERNS = [
    r"import\s+ctypes",                   # Low-level memory access  
    r"os\.system\s*\(\s*['\"]rm\s+-rf",  # Shell escape to rm -rf
    r"subprocess.*shell\s*=\s*True.*rm",  # Subprocess shell rm
    r"shutil\.rmtree\s*\(\s*['\"][/\\]", # Remove root
    r"__import__\s*\(\s*['\"]ctypes",    # Dynamic ctypes import
    r"open\s*\(\s*['\"]/(etc|proc|sys)/", # System file access (Linux)
]


class SandboxManager:
    """
    Sandboxed code execution manager.
    
    Runs code in isolated subprocesses with:
    - Timeout limits
    - Output capture and truncation
    - Dangerous command blocking
    - Full audit trail
    """

    def __init__(
        self,
        default_timeout: int = 30,
        max_output_chars: int = 5000,
        allow_network: bool = True,
    ):
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.allow_network = allow_network
        self.execution_log: List[ExecutionResult] = []

    # ─── Safety Checks ────────────────────────────────────

    def check_safety(self, code: str, language: str = "python") -> Optional[str]:
        """
        Check code for dangerous patterns.
        Returns blocking reason if unsafe, None if safe.
        """
        code_lower = code.lower().strip()

        # Check universal patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return f"Dangerous command pattern detected: {pattern}"

        # Python-specific checks
        if language == "python":
            for pattern in BLOCKED_PYTHON_PATTERNS:
                if re.search(pattern, code, re.IGNORECASE):
                    return f"Dangerous Python pattern: {pattern}"

        # Shell-specific checks
        if language in ("shell", "bash", "cmd", "powershell"):
            # Block any command that starts with these
            dangerous_starters = [
                "rm -rf /", "rm -rf ~", "del /s", "format ",
                "mkfs", "dd if=", ":(){ :", "shutdown", "reboot",
                "halt", "init 0", "init 6",
            ]
            for starter in dangerous_starters:
                if code_lower.startswith(starter):
                    return f"Blocked dangerous command: {starter}"

        return None  # Safe

    # ─── Python Execution ─────────────────────────────────

    def run_python(self, code: str, timeout: int = None) -> ExecutionResult:
        """Execute Python code in a subprocess."""
        timeout = timeout or self.default_timeout

        # Safety check first
        block_reason = self.check_safety(code, "python")
        if block_reason:
            result = ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                language="python",
                code=code,
                blocked_reason=block_reason,
            )
            self.execution_log.append(result)
            return result

        # Write code to temp file and execute
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name

            import time
            start = time.time()
            
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            elapsed = time.time() - start

            result = ExecutionResult(
                status=ExecutionStatus.SUCCESS if proc.returncode == 0 else ExecutionStatus.ERROR,
                stdout=proc.stdout[:self.max_output_chars],
                stderr=proc.stderr[:self.max_output_chars],
                return_code=proc.returncode,
                language="python",
                code=code,
                execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            result = ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                language="python",
                code=code,
                execution_time=float(timeout),
            )
        except Exception as e:
            result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=str(e),
                language="python",
                code=code,
            )
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

        self.execution_log.append(result)
        return result

    # ─── Shell Execution ──────────────────────────────────

    def run_shell(self, command: str, timeout: int = None) -> ExecutionResult:
        """Execute a shell command."""
        timeout = timeout or self.default_timeout

        # Safety check
        block_reason = self.check_safety(command, "shell")
        if block_reason:
            result = ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                language="shell",
                code=command,
                blocked_reason=block_reason,
            )
            self.execution_log.append(result)
            return result

        try:
            import time
            start = time.time()

            # Use PowerShell on Windows, bash on Unix
            if os.name == 'nt':
                shell_cmd = ["powershell", "-Command", command]
            else:
                shell_cmd = ["bash", "-c", command]

            proc = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            elapsed = time.time() - start

            result = ExecutionResult(
                status=ExecutionStatus.SUCCESS if proc.returncode == 0 else ExecutionStatus.ERROR,
                stdout=proc.stdout[:self.max_output_chars],
                stderr=proc.stderr[:self.max_output_chars],
                return_code=proc.returncode,
                language="shell",
                code=command,
                execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            result = ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                language="shell",
                code=command,
                execution_time=float(timeout),
            )
        except Exception as e:
            result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=str(e),
                language="shell",
                code=command,
            )

        self.execution_log.append(result)
        return result

    # ─── Audit ────────────────────────────────────────────

    def get_execution_history(self) -> List[ExecutionResult]:
        """Get full execution history."""
        return list(self.execution_log)

    def get_last_result(self) -> Optional[ExecutionResult]:
        """Get the most recent execution result."""
        return self.execution_log[-1] if self.execution_log else None
