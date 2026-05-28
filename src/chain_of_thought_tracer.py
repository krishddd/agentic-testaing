import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReasoningStepType(str, Enum):
    """Types of reasoning steps in the chain of thought"""
    QUERY_ANALYSIS = "query_analysis"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    RESULT_EVALUATION = "result_evaluation"
    SYNTHESIS = "synthesis"
    DECISION = "decision"
    ERROR_HANDLING = "error_handling"


class ReasoningStep(BaseModel):
    """Represents a single step in the chain of thought"""
    step_id: str
    step_number: int
    step_type: ReasoningStepType
    timestamp: str
    reasoning: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    alternatives_considered: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChainOfThoughtTrace(BaseModel):
    """Complete trace of the reasoning process"""
    trace_id: str
    query: str
    agent_type: str
    start_time: str
    end_time: Optional[str] = None
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    final_decision: Optional[str] = None
    total_reasoning_time: Optional[float] = None
    reasoning_quality_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChainOfThoughtTracer:
    """
    Tracer that captures and analyzes the agent's chain of thought process.
    Provides explainability by showing why and how decisions were made.
    """
    
    def __init__(self, trace_dir: str = "traces/chain_of_thought"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: Optional[ChainOfThoughtTrace] = None
        self.step_counter = 0
        
    def start_trace(self, query: str, agent_type: str, metadata: Dict[str, Any] = None) -> str:
        """Initialize a new chain of thought trace"""
        trace_id = f"cot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_trace = ChainOfThoughtTrace(
            trace_id=trace_id,
            query=query,
            agent_type=agent_type,
            start_time=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.step_counter = 0
        logger.info(f"Started CoT trace {trace_id} for query: {query}")
        
        return trace_id
    
    def add_reasoning_step(
        self,
        step_type: ReasoningStepType,
        reasoning: str,
        input_data: Dict[str, Any] = None,
        output_data: Dict[str, Any] = None,
        confidence_score: Optional[float] = None,
        alternatives: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a reasoning step to the current trace"""
        if not self.current_trace:
            raise ValueError("No active trace. Call start_trace() first.")
        
        self.step_counter += 1
        step_id = f"{self.current_trace.trace_id}_step_{self.step_counter}"
        
        step = ReasoningStep(
            step_id=step_id,
            step_number=self.step_counter,
            step_type=step_type,
            timestamp=datetime.now().isoformat(),
            reasoning=reasoning,
            input_data=input_data or {},
            output_data=output_data or {},
            confidence_score=confidence_score,
            alternatives_considered=alternatives or [],
            metadata=metadata or {}
        )
        
        self.current_trace.reasoning_steps.append(step)
        logger.debug(f"Added reasoning step {step_id}: {step_type.value}")
        
        return step_id
    
    def add_query_analysis(
        self,
        query_type: str,
        complexity: str,
        key_entities: List[str],
        intent: str,
        reasoning: str
    ):
        """Record query analysis reasoning"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.QUERY_ANALYSIS,
            reasoning=reasoning,
            input_data={"query": self.current_trace.query},
            output_data={
                "query_type": query_type,
                "complexity": complexity,
                "key_entities": key_entities,
                "intent": intent
            }
        )
    
    def add_tool_selection(
        self,
        selected_tool: str,
        reasoning: str,
        alternatives: List[str],
        confidence: float
    ):
        """Record tool selection reasoning"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.TOOL_SELECTION,
            reasoning=reasoning,
            output_data={"selected_tool": selected_tool},
            confidence_score=confidence,
            alternatives=alternatives
        )
    
    def add_tool_execution(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        execution_time: float,
        success: bool
    ):
        """Record tool execution details"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.TOOL_EXECUTION,
            reasoning=f"Executed {tool_name} tool",
            input_data={
                "tool_name": tool_name,
                "tool_input": tool_input
            },
            output_data={
                "tool_output": str(tool_output)[:500],  # Truncate long outputs
                "success": success,
                "execution_time": execution_time
            }
        )
    
    def add_result_evaluation(
        self,
        result: Any,
        evaluation: str,
        quality_score: float,
        reasoning: str
    ):
        """Record result evaluation reasoning"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.RESULT_EVALUATION,
            reasoning=reasoning,
            input_data={"result": str(result)[:300]},
            output_data={
                "evaluation": evaluation,
                "quality_score": quality_score
            },
            confidence_score=quality_score
        )
    
    def add_synthesis_step(
        self,
        sources: List[str],
        synthesis_method: str,
        reasoning: str,
        output: str
    ):
        """Record synthesis reasoning"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.SYNTHESIS,
            reasoning=reasoning,
            input_data={
                "sources": sources,
                "synthesis_method": synthesis_method
            },
            output_data={"synthesized_output": output[:500]}
        )
    
    def add_decision(
        self,
        decision: str,
        reasoning: str,
        confidence: float,
        alternatives: List[str] = None
    ):
        """Record decision-making reasoning"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.DECISION,
            reasoning=reasoning,
            output_data={"decision": decision},
            confidence_score=confidence,
            alternatives=alternatives or []
        )
    
    def add_error_handling(
        self,
        error: str,
        recovery_action: str,
        reasoning: str
    ):
        """Record error handling reasoning"""
        return self.add_reasoning_step(
            step_type=ReasoningStepType.ERROR_HANDLING,
            reasoning=reasoning,
            input_data={"error": error},
            output_data={"recovery_action": recovery_action}
        )
    
    def end_trace(self, final_decision: str = None) -> ChainOfThoughtTrace:
        """Complete the current trace"""
        if not self.current_trace:
            raise ValueError("No active trace to end.")
        
        self.current_trace.end_time = datetime.now().isoformat()
        self.current_trace.final_decision = final_decision
        
        # Calculate total reasoning time
        start = datetime.fromisoformat(self.current_trace.start_time)
        end = datetime.fromisoformat(self.current_trace.end_time)
        self.current_trace.total_reasoning_time = (end - start).total_seconds()
        
        # Calculate reasoning quality score
        self.current_trace.reasoning_quality_score = self._calculate_reasoning_quality()
        
        # Save trace
        self._save_trace()
        
        logger.info(f"Ended CoT trace {self.current_trace.trace_id}")
        
        trace = self.current_trace
        self.current_trace = None
        self.step_counter = 0
        
        return trace
    
    def _calculate_reasoning_quality(self) -> float:
        """Calculate overall quality score for the reasoning process"""
        if not self.current_trace or not self.current_trace.reasoning_steps:
            return 0.0
        
        # Factors: step completeness, confidence scores, error handling
        total_steps = len(self.current_trace.reasoning_steps)
        
        # Check for key reasoning types
        step_types = set(step.step_type for step in self.current_trace.reasoning_steps)
        required_types = {
            ReasoningStepType.QUERY_ANALYSIS,
            ReasoningStepType.TOOL_SELECTION,
            ReasoningStepType.SYNTHESIS
        }
        completeness = len(step_types & required_types) / len(required_types)
        
        # Average confidence score
        confidence_scores = [
            step.confidence_score for step in self.current_trace.reasoning_steps
            if step.confidence_score is not None
        ]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Error penalty
        error_steps = sum(
            1 for step in self.current_trace.reasoning_steps
            if step.step_type == ReasoningStepType.ERROR_HANDLING
        )
        error_penalty = max(0, 1 - (error_steps / total_steps))
        
        # Weighted score
        quality = (0.3 * completeness + 0.4 * avg_confidence + 0.3 * error_penalty)
        
        return round(quality, 3)
    
    def _save_trace(self):
        """Save the current trace to disk"""
        if not self.current_trace:
            return
        
        filename = self.trace_dir / f"{self.current_trace.trace_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                self.current_trace.model_dump(), # Changed .dict() to .model_dump()
                f,
                indent=2,
                default=str
            )
        
        logger.info(f"Saved CoT trace to {filename}")
    
    def get_trace_summary(self, trace_id: str = None) -> Dict[str, Any]:
        """Get a summary of a trace"""
        if trace_id:
            trace_file = self.trace_dir / f"{trace_id}.json"
            if not trace_file.exists():
                raise ValueError(f"Trace {trace_id} not found")
            
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
                trace = ChainOfThoughtTrace(**trace_data)
        else:
            if not self.current_trace:
                raise ValueError("No active trace")
            trace = self.current_trace
        
        step_type_counts = {}
        for step in trace.reasoning_steps:
            step_type_counts[step.step_type.value] = step_type_counts.get(step.step_type.value, 0) + 1
        
        return {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "agent_type": trace.agent_type,
            "total_steps": len(trace.reasoning_steps),
            "step_type_distribution": step_type_counts,
            "reasoning_time": trace.total_reasoning_time,
            "quality_score": trace.reasoning_quality_score,
            "final_decision": trace.final_decision
        }
    
    def visualize_trace(self, trace_id: str = None) -> str:
        """Generate a human-readable visualization of the reasoning chain"""
        if trace_id:
            trace_file = self.trace_dir / f"{trace_id}.json"
            if not trace_file.exists():
                raise ValueError(f"Trace {trace_id} not found")
            
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
                trace = ChainOfThoughtTrace(**trace_data)
        else:
            if not self.current_trace:
                raise ValueError("No active trace")
            trace = self.current_trace
        
        lines = [
            f"=== Chain of Thought Trace: {trace.trace_id} ===",
            f"Query: {trace.query}",
            f"Agent: {trace.agent_type}",
            f"Quality Score: {trace.reasoning_quality_score:.2f}",
            "",
            "Reasoning Steps:",
            ""
        ]
        
        for step in trace.reasoning_steps:
            lines.append(f"Step {step.step_number}: [{step.step_type.value.upper()}]")
            lines.append(f"  Reasoning: {step.reasoning}")
            
            if step.confidence_score is not None:
                lines.append(f"  Confidence: {step.confidence_score:.2f}")
            
            if step.alternatives_considered:
                lines.append(f"  Alternatives: {', '.join(step.alternatives_considered)}")
            
            if step.output_data:
                lines.append(f"  Output: {json.dumps(step.output_data, indent=4)[:200]}...")
            
            lines.append("")
        
        if trace.final_decision:
            lines.append(f"Final Decision: {trace.final_decision}")
        
        return "\n".join(lines)
    
    def export_trace_for_analysis(self, trace_id: str) -> Dict[str, Any]:
        """Export trace in a format suitable for analysis"""
        trace_file = self.trace_dir / f"{trace_id}.json"
        
        if not trace_file.exists():
            raise ValueError(f"Trace {trace_id} not found")
        
        with open(trace_file, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        
        return trace_data