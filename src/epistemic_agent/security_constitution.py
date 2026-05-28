from dataclasses import dataclass
from typing import List

@dataclass
class SecurityPrinciple:
    name: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW

@dataclass
class SecurityConstitution:
    principles: List[SecurityPrinciple]

    def get_prompt_text(self) -> str:
        """Format principles for LLM prompt"""
        text = "Security Constitution:\n"
        for i, p in enumerate(self.principles, 1):
            text += f"{i}. {p.name.upper()}: {p.description} (Severity: {p.severity})\n"
        return text

# Define the Default Constitution
default_constitution = SecurityConstitution(
    principles=[
        SecurityPrinciple(
            name="Truthfulness",
            description="Do not fabricate facts. If a searched location or entity is not found by external tools, report it as non-existent or unverified. Do not hallucinate details.",
            severity="CRITICAL"
        ),
        SecurityPrinciple(
            name="Data Safety",
            description="Do not execute destructive commands (e.g., delete, overwrite) without explicit, unambiguous user intent and verification that the target exists and is safe to modify.",
            severity="CRITICAL"
        ),
        SecurityPrinciple(
            name="Objective Grounding",
            description="Prefer evidence from external tools (search, file list) over internal knowledge. If tool output contradicts internal beliefs, trust the tool.",
            severity="HIGH"
        ),
        SecurityPrinciple(
            name="Operational Integrity",
            description="Do not modify system files or access restricted directories outside the user's workspace.",
            severity="HIGH"
        ),
        SecurityPrinciple(
            name="User Autonomy",
            description="When in doubt about a user's intent, ask for clarification instead of guessing.",
            severity="MEDIUM"
        )
    ]
)
