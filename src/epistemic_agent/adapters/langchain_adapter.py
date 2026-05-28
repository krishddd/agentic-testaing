"""
LangChain/CrewAI Adapter for Epistemic Security

Provides callback handlers and tool wrappers for integrating epistemic
security with LangChain and CrewAI agents.

Usage:
    from langchain_openai import ChatOpenAI
    from epistemic_agent.adapters import EpistemicCallbackHandler, EpistemicToolWrapper
    
    # Method 1: Callback Handler
    llm = ChatOpenAI(callbacks=[EpistemicCallbackHandler()])
    
    # Method 2: Tool Wrapper
    @EpistemicToolWrapper
    def delete_file(filepath: str) -> str:
        ...
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import LangChain
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult
    from langchain_core.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-core not installed")
    
    # Create stub classes
    class BaseCallbackHandler:
        pass
    
    class BaseTool:
        pass

from ..toolgate import ToolGate
from ..policy_enforcer import PolicyEnforcer, VerdictType
from ..active_security_monitor import ActiveSecurityMonitor
from ..generative_model import GenerativeModel, Action, ActionType


@dataclass
class EpistemicEvent:
    """Event from epistemic security layer"""
    event_type: str  # "tool_check", "response_check", "belief_update"
    tool_name: Optional[str] = None
    allowed: bool = True
    reason: str = ""
    surprisal: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EpistemicCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for epistemic security integration.
    
    Intercepts:
    - Tool invocations (pre-check)
    - LLM responses (hallucination check)
    - Agent actions (policy audit)
    
    Example:
        from langchain_openai import ChatOpenAI
        
        handler = EpistemicCallbackHandler(
            on_blocked=lambda e: print(f"Blocked: {e.reason}"),
            block_on_failure=True
        )
        
        llm = ChatOpenAI(callbacks=[handler])
    """
    
    def __init__(
        self,
        tool_gate: Optional[ToolGate] = None,
        policy_enforcer: Optional[PolicyEnforcer] = None,
        security_monitor: Optional[ActiveSecurityMonitor] = None,
        on_event: Optional[Callable[[EpistemicEvent], None]] = None,
        on_blocked: Optional[Callable[[EpistemicEvent], None]] = None,
        block_on_failure: bool = False,
        log_events: bool = True
    ):
        """
        Initialize the callback handler.
        
        Args:
            tool_gate: ToolGate instance (created if None)
            policy_enforcer: PolicyEnforcer instance (created if None)
            security_monitor: SecurityMonitor instance (created if None)
            on_event: Callback for all epistemic events
            on_blocked: Callback when action is blocked
            block_on_failure: If True, raise exception on blocked actions
            log_events: If True, log all events
        """
        super().__init__()
        
        self.tool_gate = tool_gate or ToolGate()
        self.policy_enforcer = policy_enforcer or PolicyEnforcer()
        self.security_monitor = security_monitor or ActiveSecurityMonitor()
        
        self.model = GenerativeModel()
        self.model.initialize_belief()
        
        self.on_event = on_event
        self.on_blocked = on_blocked
        self.block_on_failure = block_on_failure
        self.log_events = log_events
        
        self._action_history: List[str] = []
        self._events: List[EpistemicEvent] = []
        self._current_context: str = ""
    
    @property
    def events(self) -> List[EpistemicEvent]:
        """Get all recorded events"""
        return self._events.copy()
    
    def clear_events(self):
        """Clear recorded events"""
        self._events.clear()
        self._action_history.clear()
    
    def _emit_event(self, event: EpistemicEvent):
        """Emit an epistemic event"""
        self._events.append(event)
        
        if self.log_events:
            log_msg = f"[Epistemic] {event.event_type}: allowed={event.allowed}"
            if event.tool_name:
                log_msg += f", tool={event.tool_name}"
            if event.reason:
                log_msg += f", reason={event.reason}"
            logger.info(log_msg)
        
        if self.on_event:
            self.on_event(event)
        
        if not event.allowed and self.on_blocked:
            self.on_blocked(event)
        
        if not event.allowed and self.block_on_failure:
            raise SecurityBlockedError(
                f"Action blocked by epistemic security: {event.reason}"
            )
    
    # LangChain Callback Methods
    
    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Called when LLM starts processing"""
        if prompts:
            self._current_context = prompts[0][:500]
    
    def on_llm_end(self, response: Any, **kwargs):
        """Called when LLM finishes - check for hallucination"""
        try:
            # Extract text from response
            if hasattr(response, 'generations') and response.generations:
                text = response.generations[0][0].text if response.generations[0] else ""
            else:
                text = str(response)
            
            if text:
                surprisal, is_hallucination = self.security_monitor.detect_hallucination(
                    prediction=f"Response to: {self._current_context[:100]}",
                    observation=text[:500]
                )
                
                event = EpistemicEvent(
                    event_type="response_check",
                    allowed=not is_hallucination,
                    surprisal=surprisal,
                    reason="High semantic divergence" if is_hallucination else "OK",
                    metadata={"text_preview": text[:100]}
                )
                
                self._emit_event(event)
                
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
    
    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        """Called before tool execution - verify safety"""
        tool_name = serialized.get("name", "unknown")
        
        # Parse input
        try:
            tool_input = json.loads(input_str) if isinstance(input_str, str) else input_str
        except json.JSONDecodeError:
            tool_input = {"raw": input_str}
        
        # Create action
        action = Action(
            name=tool_name,
            action_type=ActionType.PRAGMATIC,
            arguments=tool_input if isinstance(tool_input, dict) else {},
            description=self._current_context
        )
        
        # ToolGate check
        gate_ok, gate_msg = self.tool_gate.verify_preconditions(action, self.model.current_belief)
        
        # Policy check
        audit = self.policy_enforcer.audit_action(action, self._current_context)
        
        # Track
        self._action_history.append(tool_name)
        
        # Create event
        allowed = gate_ok and audit.verdict == VerdictType.ALLOWED
        reason = gate_msg if not gate_ok else (audit.reasoning if not allowed else "OK")
        
        event = EpistemicEvent(
            event_type="tool_check",
            tool_name=tool_name,
            allowed=allowed,
            reason=reason,
            metadata={
                "input": tool_input,
                "verdict": audit.verdict.value
            }
        )
        
        self._emit_event(event)
    
    def on_tool_end(self, output: str, **kwargs):
        """Called after tool execution"""
        # Could verify postconditions here
        pass
    
    def on_agent_action(self, action: Any, **kwargs):
        """Called when agent decides on an action"""
        if hasattr(action, 'tool'):
            # This is an AgentAction
            tool_name = action.tool
            tool_input = action.tool_input
            
            action_obj = Action(
                name=tool_name,
                action_type=ActionType.PRAGMATIC,
                arguments=tool_input if isinstance(tool_input, dict) else {},
                description=str(action.log) if hasattr(action, 'log') else ""
            )
            
            audit = self.policy_enforcer.audit_action(action_obj, self._current_context)
            
            event = EpistemicEvent(
                event_type="agent_action",
                tool_name=tool_name,
                allowed=audit.verdict == VerdictType.ALLOWED,
                reason=audit.reasoning,
                metadata={"verdict": audit.verdict.value}
            )
            
            self._emit_event(event)
    
    def on_agent_finish(self, finish: Any, **kwargs):
        """Called when agent finishes"""
        pass


class SecurityBlockedError(Exception):
    """Raised when an action is blocked by epistemic security"""
    pass


class EpistemicToolWrapper:
    """
    Decorator/wrapper for LangChain tools with epistemic security.
    
    Usage:
        @EpistemicToolWrapper
        def delete_file(filepath: str) -> str:
            '''Delete a file'''
            os.remove(filepath)
            return f"Deleted {filepath}"
        
        # Or wrap existing tool
        secured_tool = EpistemicToolWrapper(existing_tool)
    """
    
    def __init__(
        self,
        func_or_tool: Union[Callable, Any] = None,
        tool_gate: Optional[ToolGate] = None,
        policy_enforcer: Optional[PolicyEnforcer] = None,
        block_on_failure: bool = True
    ):
        self.tool_gate = tool_gate or ToolGate()
        self.policy_enforcer = policy_enforcer or PolicyEnforcer()
        self.block_on_failure = block_on_failure
        
        self.model = GenerativeModel()
        self.model.initialize_belief()
        
        self._func = None
        self._tool = None
        
        if func_or_tool is not None:
            if callable(func_or_tool) and not isinstance(func_or_tool, type):
                # It's a function
                self._func = func_or_tool
            else:
                # It's a tool instance
                self._tool = func_or_tool
    
    def __call__(self, *args, **kwargs):
        if self._func is None:
            # Being used as decorator
            func = args[0]
            wrapper = EpistemicToolWrapper(
                func,
                tool_gate=self.tool_gate,
                policy_enforcer=self.policy_enforcer,
                block_on_failure=self.block_on_failure
            )
            return wrapper._create_wrapped_func()
        else:
            # Being called as function
            return self._execute(*args, **kwargs)
    
    def _create_wrapped_func(self):
        """Create a wrapped version of the function"""
        @wraps(self._func)
        def wrapped(*args, **kwargs):
            return self._execute(*args, **kwargs)
        return wrapped
    
    def _execute(self, *args, **kwargs):
        """Execute with security checks"""
        tool_name = self._func.__name__ if self._func else "unknown"
        
        # Build tool input from args/kwargs
        tool_input = dict(kwargs)
        if args:
            tool_input["_args"] = args
        
        # Create action
        action = Action(
            name=tool_name,
            action_type=ActionType.PRAGMATIC,
            arguments=tool_input,
            description=self._func.__doc__ if self._func else ""
        )
        
        # Verify
        gate_ok, gate_msg = self.tool_gate.verify_preconditions(action, self.model.current_belief)
        audit = self.policy_enforcer.audit_action(action, "")
        
        allowed = gate_ok and audit.verdict == VerdictType.ALLOWED
        
        if not allowed and self.block_on_failure:
            reason = gate_msg if not gate_ok else audit.reasoning
            raise SecurityBlockedError(f"Tool {tool_name} blocked: {reason}")
        
        # Execute
        if self._func:
            return self._func(*args, **kwargs)
        elif self._tool and hasattr(self._tool, 'run'):
            return self._tool.run(*args, **kwargs)
        else:
            raise ValueError("No function or tool to execute")


def create_epistemic_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Any:
    """
    Create a LangChain tool with epistemic security.
    
    Args:
        func: The function to wrap
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        
    Returns:
        A secured LangChain tool
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("langchain-core required. Install: pip install langchain-core")
    
    from langchain_core.tools import StructuredTool
    
    # Wrap with security
    secured_func = EpistemicToolWrapper(func)
    
    return StructuredTool.from_function(
        func=secured_func,
        name=name or func.__name__,
        description=description or func.__doc__ or ""
    )


# CrewAI specific adapter
class EpistemicCrewAICallback:
    """
    Callback class for CrewAI integration.
    
    Usage with CrewAI:
        from crewai import Crew, Agent, Task
        
        callback = EpistemicCrewAICallback()
        
        crew = Crew(
            agents=[...],
            tasks=[...],
            step_callback=callback.on_step
        )
    """
    
    def __init__(self):
        self.tool_gate = ToolGate()
        self.policy_enforcer = PolicyEnforcer()
        self.security_monitor = ActiveSecurityMonitor()
        
        self.model = GenerativeModel()
        self.model.initialize_belief()
        
        self.events: List[Dict] = []
    
    def on_step(self, step_output: Any) -> Any:
        """
        Called on each CrewAI step.
        
        Args:
            step_output: Output from the current step
            
        Returns:
            Modified step output (or original if allowed)
        """
        # Extract tool info if present
        if hasattr(step_output, 'tool') and step_output.tool:
            tool_name = step_output.tool
            tool_input = getattr(step_output, 'tool_input', {})
            
            action = Action(
                name=tool_name,
                action_type=ActionType.PRAGMATIC,
                arguments=tool_input if isinstance(tool_input, dict) else {},
                description=""
            )
            
            gate_ok, gate_msg = self.tool_gate.verify_preconditions(action, self.model.current_belief)
            audit = self.policy_enforcer.audit_action(action, "")
            
            allowed = gate_ok and audit.verdict == VerdictType.ALLOWED
            
            self.events.append({
                "type": "tool_check",
                "tool": tool_name,
                "allowed": allowed,
                "reason": gate_msg if not gate_ok else audit.reasoning
            })
            
            if not allowed:
                logger.warning(f"CrewAI tool {tool_name} blocked by epistemic security")
        
        return step_output
    
    def get_summary(self) -> Dict:
        """Get summary of security events"""
        blocked = [e for e in self.events if not e.get("allowed", True)]
        return {
            "total_events": len(self.events),
            "blocked_count": len(blocked),
            "blocked_tools": [e.get("tool") for e in blocked]
        }
