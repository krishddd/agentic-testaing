"""
Claude/Anthropic Adapter for Epistemic Security

Wraps Claude API calls with epistemic security layer for:
- Tool call interception and verification
- Response hallucination detection
- Constitutional audit integration

Usage:
    from epistemic_agent.adapters import EpistemicClaudeClient
    
    client = EpistemicClaudeClient()
    response = await client.chat(
        messages=[{"role": "user", "content": "Delete the temp files"}],
        tools=[{"name": "delete_file", ...}]
    )
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import anthropic
try:
    import anthropic
    from anthropic.types import Message, ContentBlock, ToolUseBlock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not installed")

from ..toolgate import ToolGate
from ..policy_enforcer import PolicyEnforcer, VerdictType
from ..active_security_monitor import ActiveSecurityMonitor
from ..generative_model import GenerativeModel, Action, ActionType


@dataclass
class SecurityResult:
    """Result of security check"""
    allowed: bool
    reason: str
    risk_level: str
    surprisal: float = 0.0


class ClaudeAdapter:
    """
    Adapter that intercepts Claude tool calls for epistemic security checks.
    
    Can be used as:
    1. A wrapper around tool execution
    2. A pre-check before tool calls
    3. A post-check for response validation
    """
    
    def __init__(self):
        self.tool_gate = ToolGate()
        self.policy_enforcer = PolicyEnforcer()
        self.security_monitor = ActiveSecurityMonitor()
        self.model = GenerativeModel()
        self.model.initialize_belief()
        
        self._action_history: List[str] = []
    
    def verify_tool_call(
        self,
        tool_name: str,
        tool_input: Dict,
        context: str = ""
    ) -> SecurityResult:
        """
        Verify a tool call before execution.
        
        Args:
            tool_name: Name of the tool being called
            tool_input: Arguments to the tool
            context: Context about why the tool is being called
            
        Returns:
            SecurityResult indicating if the call is allowed
        """
        # Create action
        action = Action(
            name=tool_name,
            action_type=ActionType.PRAGMATIC,
            arguments=tool_input,
            description=context
        )
        
        # ToolGate check
        gate_ok, gate_msg = self.tool_gate.verify_preconditions(action, self.model.current_belief)
        
        # Policy check
        audit = self.policy_enforcer.audit_action(action, context)
        
        # Track action
        self._action_history.append(tool_name)
        
        # Determine result
        allowed = gate_ok and audit.verdict == VerdictType.ALLOWED
        
        reason = ""
        if not gate_ok:
            reason = gate_msg
        elif audit.verdict != VerdictType.ALLOWED:
            reason = audit.reasoning
        
        risk_level = "low"
        if audit.verdict == VerdictType.BLOCKED:
            risk_level = "high"
        elif audit.verdict == VerdictType.NEEDS_CLARIFICATION:
            risk_level = "medium"
        
        return SecurityResult(
            allowed=allowed,
            reason=reason,
            risk_level=risk_level
        )
    
    def check_response(self, predicted: str, actual: str) -> SecurityResult:
        """
        Check a response for hallucination.
        
        Args:
            predicted: What was expected
            actual: What was actually received
            
        Returns:
            SecurityResult with hallucination assessment
        """
        surprisal, is_hallucination = self.security_monitor.detect_hallucination(
            prediction=predicted,
            observation=actual
        )
        
        return SecurityResult(
            allowed=not is_hallucination,
            reason="High semantic divergence detected" if is_hallucination else "Response appears consistent",
            risk_level="high" if is_hallucination else "low",
            surprisal=surprisal
        )
    
    def wrap_tool_executor(
        self,
        executor: Callable[[str, Dict], Any]
    ) -> Callable[[str, Dict], Any]:
        """
        Wrap a tool executor function with security checks.
        
        Args:
            executor: Function that executes tools (name, input) -> result
            
        Returns:
            Wrapped executor with security checks
        """
        def wrapped_executor(tool_name: str, tool_input: Dict) -> Any:
            # Pre-check
            result = self.verify_tool_call(tool_name, tool_input)
            
            if not result.allowed:
                return {
                    "error": f"Tool call blocked by security layer: {result.reason}",
                    "risk_level": result.risk_level
                }
            
            # Execute
            output = executor(tool_name, tool_input)
            
            # Post-check (verify postconditions)
            action = Action(
                name=tool_name,
                action_type=ActionType.PRAGMATIC,
                arguments=tool_input,
                description=""
            )
            
            post_ok, post_msg = self.tool_gate.verify_postconditions(
                action, str(output), self.model.current_belief
            )
            
            if not post_ok:
                logger.warning(f"Postcondition failed for {tool_name}: {post_msg}")
            
            return output
        
        return wrapped_executor


class EpistemicClaudeClient:
    """
    Claude client with integrated epistemic security.
    
    Drop-in replacement for anthropic.Anthropic with security features.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.adapter = ClaudeAdapter()
        
        # Tool execution handlers
        self._tool_handlers: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, handler: Callable[[Dict], Any]):
        """Register a tool handler"""
        self._tool_handlers[name] = handler
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Send a chat request with epistemic security.
        
        Returns dict with:
        - response: Claude's response
        - security: Security check results
        - tool_results: Results of any tool calls
        """
        # Build request
        request_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        
        if system:
            request_kwargs["system"] = system
        if tools:
            request_kwargs["tools"] = tools
        
        request_kwargs.update(kwargs)
        
        # Call Claude
        response = await asyncio.to_thread(
            self.client.messages.create,
            **request_kwargs
        )
        
        # Process response
        result = {
            "response": response,
            "security": [],
            "tool_results": []
        }
        
        # Handle tool use blocks
        for block in response.content:
            if hasattr(block, 'type') and block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                
                # Security check
                security_result = self.adapter.verify_tool_call(
                    tool_name,
                    tool_input,
                    context=messages[-1].get("content", "") if messages else ""
                )
                
                result["security"].append({
                    "tool": tool_name,
                    "allowed": security_result.allowed,
                    "reason": security_result.reason,
                    "risk_level": security_result.risk_level
                })
                
                # Execute if allowed and handler exists
                if security_result.allowed and tool_name in self._tool_handlers:
                    try:
                        tool_output = self._tool_handlers[tool_name](tool_input)
                        result["tool_results"].append({
                            "tool": tool_name,
                            "output": tool_output,
                            "success": True
                        })
                    except Exception as e:
                        result["tool_results"].append({
                            "tool": tool_name,
                            "error": str(e),
                            "success": False
                        })
                elif not security_result.allowed:
                    result["tool_results"].append({
                        "tool": tool_name,
                        "blocked": True,
                        "reason": security_result.reason
                    })
        
        # Check final response for hallucination
        text_content = "".join(
            block.text for block in response.content 
            if hasattr(block, 'text')
        )
        
        if text_content:
            expected = f"Response to: {messages[-1].get('content', '')[:100]}" if messages else ""
            hallucination_check = self.adapter.check_response(expected, text_content)
            
            result["security"].append({
                "type": "hallucination_check",
                "surprisal": hallucination_check.surprisal,
                "is_risk": not hallucination_check.allowed,
                "reason": hallucination_check.reason
            })
        
        return result
    
    def with_tool_loop(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        max_iterations: int = 10,
        **kwargs
    ):
        """
        Run a tool-use loop with security checks at each step.
        
        Continues until Claude stops requesting tools or max iterations reached.
        """
        async def _loop():
            current_messages = list(messages)
            iterations = 0
            all_results = []
            
            while iterations < max_iterations:
                result = await self.chat(
                    messages=current_messages,
                    tools=tools,
                    **kwargs
                )
                
                all_results.append(result)
                
                response = result["response"]
                
                # Check if we need to continue (tool use requested)
                tool_uses = [
                    b for b in response.content 
                    if hasattr(b, 'type') and b.type == "tool_use"
                ]
                
                if not tool_uses:
                    break
                
                # Add assistant message
                current_messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool results
                for tool_result in result["tool_results"]:
                    current_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_uses[0].id,  # Simplified
                            "content": json.dumps(tool_result)
                        }]
                    })
                
                iterations += 1
            
            return {
                "iterations": iterations,
                "results": all_results,
                "final_response": all_results[-1] if all_results else None
            }
        
        return asyncio.run(_loop())


# Convenience functions

def create_secured_claude_client(api_key: Optional[str] = None) -> EpistemicClaudeClient:
    """Create a Claude client with epistemic security"""
    return EpistemicClaudeClient(api_key=api_key)


def verify_tool_call(tool_name: str, tool_input: Dict, context: str = "") -> SecurityResult:
    """One-shot tool verification"""
    adapter = ClaudeAdapter()
    return adapter.verify_tool_call(tool_name, tool_input, context)
