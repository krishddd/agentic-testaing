"""
Code Interpreter — LLM-powered code generation + sandboxed execution.

Follows Open Interpreter's pattern:
  User request -> LLM generates code -> Safety check -> Execute -> Capture output -> Return

Integrated with the Active Inference agent's belief framework so that
code execution actions go through the same EFE/policy pipeline.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .sandbox import SandboxManager, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Code generated from a natural language request."""
    language: str
    code: str
    explanation: str
    task_description: str
    requires_confirmation: bool = True


class CodeInterpreter:
    """
    Sandboxed code execution engine.
    
    Capabilities:
    - Execute Python code
    - Execute shell commands
    - Auto-detect language from code blocks
    - Self-healing: if code fails, can refine and retry
    - Full execution history
    """

    def __init__(
        self,
        allowed_languages: List[str] = None,
        default_timeout: int = 30,
        max_retries: int = 2,
    ):
        self.allowed_languages = allowed_languages or ["python", "shell"]
        self.sandbox = SandboxManager(default_timeout=default_timeout)
        self.max_retries = max_retries
        self.history: List[Dict[str, Any]] = []

    # --- Language Detection -------------------------------

    @staticmethod
    def detect_language(code: str) -> str:
        """Auto-detect language from code content."""
        code_stripped = code.strip()

        # Check for markdown code blocks
        block_match = re.match(r"```(\w+)\n", code_stripped)
        if block_match:
            lang = block_match.group(1).lower()
            lang_map = {
                "python": "python", "py": "python", "python3": "python",
                "bash": "shell", "sh": "shell", "shell": "shell",
                "cmd": "shell", "powershell": "shell", "ps1": "shell",
            }
            return lang_map.get(lang, "python")

        # Heuristic: shell commands
        shell_indicators = [
            "ls ", "dir ", "cd ", "mkdir ", "cp ", "mv ", "rm ",
            "echo ", "cat ", "grep ", "find ", "pip ", "npm ",
            "git ", "docker ", "curl ", "wget ",
        ]
        if any(code_stripped.lower().startswith(ind) for ind in shell_indicators):
            return "shell"

        # Default to Python
        return "python"

    @staticmethod
    def extract_code(code: str) -> str:
        """Extract code from markdown code blocks if present."""
        # Match ```lang\n...\n```
        match = re.search(r"```\w*\n(.*?)```", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.strip()

    # --- Execute ------------------------------------------

    def execute(
        self,
        code: str,
        language: str = None,
        timeout: int = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox.
        
        Auto-detects language if not specified.
        Extracts code from markdown blocks if present.
        """
        # Clean and detect
        clean_code = self.extract_code(code)
        lang = language or self.detect_language(code)

        if lang not in self.allowed_languages:
            result = ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                language=lang,
                code=clean_code,
                blocked_reason=f"Language '{lang}' not in allowed list: {self.allowed_languages}",
            )
            self._log_execution(clean_code, lang, result)
            return result

        # Execute
        if lang == "python":
            result = self.sandbox.run_python(clean_code, timeout=timeout)
        elif lang in ("shell", "bash", "cmd", "powershell"):
            result = self.sandbox.run_shell(clean_code, timeout=timeout)
        else:
            result = ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                language=lang,
                code=clean_code,
                blocked_reason=f"No executor for language: {lang}",
            )

        self._log_execution(clean_code, lang, result)
        return result

    def execute_with_retry(
        self,
        code: str,
        language: str = None,
        timeout: int = None,
    ) -> ExecutionResult:
        """Execute code with automatic retry on failure."""
        result = self.execute(code, language, timeout)
        
        # If blocked, don't retry
        if result.status == ExecutionStatus.BLOCKED:
            return result

        # If error, attempt basic fixes and retry
        retries = 0
        while result.status == ExecutionStatus.ERROR and retries < self.max_retries:
            retries += 1
            logger.info(f"[CodeInterpreter] Retry {retries}/{self.max_retries}")
            
            # Basic auto-fix: if missing import, add common ones
            fixed_code = self._attempt_auto_fix(code, result.stderr)
            if fixed_code != code:
                code = fixed_code
                result = self.execute(code, language, timeout)
            else:
                break  # No fix found

        return result

    # --- Auto-fix -----------------------------------------

    @staticmethod
    def _attempt_auto_fix(code: str, error: str) -> str:
        """Attempt basic fixes for common Python errors."""
        if not error:
            return code

        # Fix: ModuleNotFoundError
        module_match = re.search(r"No module named '(\w+)'", error)
        if module_match:
            module = module_match.group(1)
            # Try adding a pip install
            return f"import subprocess\nsubprocess.run(['pip', 'install', '{module}'], capture_output=True)\n{code}"

        # Fix: NameError (undefined variable)
        name_match = re.search(r"name '(\w+)' is not defined", error)
        if name_match:
            name = name_match.group(1)
            # Common imports
            auto_imports = {
                "os": "import os",
                "sys": "import sys",
                "json": "import json",
                "re": "import re",
                "math": "import math",
                "datetime": "from datetime import datetime",
                "Path": "from pathlib import Path",
                "np": "import numpy as np",
                "pd": "import pandas as pd",
            }
            if name in auto_imports:
                return f"{auto_imports[name]}\n{code}"

        return code  # No fix

    # --- History ------------------------------------------

    def _log_execution(self, code: str, language: str, result: ExecutionResult):
        """Record execution in history."""
        self.history.append({
            "code": code[:200],
            "language": language,
            "status": result.status.value,
            "output_preview": str(result)[:100],
            "timestamp": result.timestamp,
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def get_capabilities(self) -> Dict[str, Any]:
        """Describe what this interpreter can do."""
        return {
            "languages": self.allowed_languages,
            "timeout": self.sandbox.default_timeout,
            "max_retries": self.max_retries,
            "safety": "Dangerous commands blocked (30+ patterns)",
            "features": [
                "Python code execution",
                "Shell/PowerShell commands",
                "Auto language detection",
                "Markdown code block extraction",
                "Auto-retry with basic error fixing",
                "Full execution audit trail",
            ],
        }
