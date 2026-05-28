import time
import logging
from typing import Dict, Any, Optional, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

from src.chain_of_thought_tracer import (
    ChainOfThoughtTracer,
    ReasoningStepType
)
from src.tool_attribution import ToolAttributionTracker

logger = logging.getLogger(__name__)


class ExplainabilityCallback(BaseCallbackHandler):
    """
    Unified callback handler that captures both chain-of-thought reasoning
    and tool attribution in real-time during agent execution.
    """
    
    def __init__(
        self,
        query: str,
        agent_type: str,
        cot_tracer: ChainOfThoughtTracer,
        attribution_tracker: ToolAttributionTracker
    ):
        self.query = query
        self.agent_type = agent_type
        self.cot_tracer = cot_tracer
        self.attribution_tracker = attribution_tracker
        
        # Tracking state
        self.current_tool_start_time: Optional[float] = None
        self.current_tool_name: Optional[str] = None
        self.current_tool_input: Optional[Dict] = None
        self.llm_calls: List[str] = []
        self.trace_initialized = False  # Add flag to track initialization
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when the agent chain starts"""
        # Skip if already initialized or if serialized is None
        if self.trace_initialized or serialized is None:
            return
            
        try:
            logger.info("Agent chain started - initializing explainability tracking")
            
            # Start both tracers
            self.cot_tracer.start_trace(
                query=self.query,
                agent_type=self.agent_type,
                metadata={"chain_type": serialized.get("name", "unknown") if serialized else "unknown"}
            )
            
            self.attribution_tracker.start_tracking(
                query=self.query,
                metadata={"agent_type": self.agent_type}
            )
            
            # Add initial query analysis
            self.cot_tracer.add_query_analysis(
                query_type=self._classify_query_type(self.query),
                complexity=self._assess_complexity(self.query),
                key_entities=self._extract_entities(self.query),
                intent=self._determine_intent(self.query),
                reasoning=f"Analyzing query: '{self.query}'"
            )
            
            self.trace_initialized = True
        except Exception as e:
            logger.error(f"Error initializing explainability tracking: {e}", exc_info=True)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM is invoked"""
        try:
            logger.debug(f"LLM invoked with {len(prompts)} prompts")
            self.llm_calls.extend(prompts)
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM finishes"""
        try:
            logger.debug("LLM call completed")
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
    
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Called when agent decides to use a tool"""
        try:
            self.current_tool_name = action.tool
            self.current_tool_input = action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input}
            self.current_tool_start_time = time.time()
            
            # Extract reasoning from the agent's log
            reasoning = action.log.strip() if action.log else f"Decided to use {action.tool}"
            
            # Add to chain of thought
            self.cot_tracer.add_tool_selection(
                selected_tool=action.tool,
                reasoning=reasoning,
                alternatives=self._extract_alternatives(reasoning),
                confidence=self._estimate_confidence(reasoning)
            )
            
            logger.info(f"Agent selected tool: {action.tool}")
        except Exception as e:
            logger.error(f"Error in on_agent_action: {e}", exc_info=True)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when tool execution starts"""
        try:
            tool_name = serialized.get('name', 'unknown') if serialized else 'unknown'
            logger.debug(f"Tool execution started: {tool_name}")
        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}")
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when tool execution completes"""
        if not self.current_tool_name or self.current_tool_start_time is None:
            logger.warning("Tool ended but no start tracked")
            return
        
        try:
            execution_time = time.time() - self.current_tool_start_time
            
            # Determine tool type
            tool_type = self._categorize_tool(self.current_tool_name)
            
            # Track tool execution in CoT
            self.cot_tracer.add_tool_execution(
                tool_name=self.current_tool_name,
                tool_input=self.current_tool_input or {},
                tool_output=output,
                execution_time=execution_time,
                success=True
            )
            
            # Track in attribution
            contribution_score = self._calculate_contribution_score(
                self.current_tool_name,
                output
            )
            
            self.attribution_tracker.track_tool_usage(
                tool_name=self.current_tool_name,
                tool_type=tool_type,
                tool_input=self.current_tool_input or {},
                tool_output=output,
                execution_duration=execution_time,
                contribution_score=contribution_score,
                output_used=True
            )
            
            # Add source attribution if applicable
            self._add_source_attribution(self.current_tool_name, output)
            
            # Evaluate the result
            evaluation = self._evaluate_tool_output(output)
            self.cot_tracer.add_result_evaluation(
                result=output,
                evaluation=evaluation["assessment"],
                quality_score=evaluation["quality_score"],
                reasoning=evaluation["reasoning"]
            )
            
            logger.info(f"Tool '{self.current_tool_name}' completed in {execution_time:.2f}s")
        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}", exc_info=True)
        finally:
            # Reset tracking state
            self.current_tool_name = None
            self.current_tool_input = None
            self.current_tool_start_time = None
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called when tool execution fails"""
        if not self.current_tool_name:
            return
        
        try:
            execution_time = time.time() - self.current_tool_start_time if self.current_tool_start_time else 0.0
            error_msg = str(error)
            
            # Track error in CoT
            self.cot_tracer.add_error_handling(
                error=error_msg,
                recovery_action="Attempting fallback or alternative approach",
                reasoning=f"Tool {self.current_tool_name} failed: {error_msg}"
            )
            
            # Track in attribution with error flag
            self.attribution_tracker.track_tool_usage(
                tool_name=self.current_tool_name,
                tool_type=self._categorize_tool(self.current_tool_name),
                tool_input=self.current_tool_input or {},
                tool_output=f"Error: {error_msg}",
                execution_duration=execution_time,
                contribution_score=0.0,
                output_used=False,
                error=error_msg
            )
            
            logger.error(f"Tool error: {self.current_tool_name} - {error_msg}")
        except Exception as e:
            logger.error(f"Error in on_tool_error: {e}", exc_info=True)
        finally:
            # Reset state
            self.current_tool_name = None
            self.current_tool_input = None
            self.current_tool_start_time = None
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Called when agent produces final answer"""
        try:
            final_output = finish.return_values.get("output", "")
            
            # Add synthesis step to CoT
            sources_used = [
                contrib.tool_name
                for contrib in self.attribution_tracker.tool_contributions
                if contrib.output_used_in_final
            ]
            
            self.cot_tracer.add_synthesis_step(
                sources=sources_used,
                synthesis_method="LLM-based synthesis",
                reasoning="Combining information from all sources to generate final answer",
                output=final_output
            )
            
            # Add final decision
            self.cot_tracer.add_decision(
                decision="Provide final answer to user",
                reasoning="All necessary information has been gathered and synthesized",
                confidence=self._estimate_answer_confidence(final_output)
            )
            
            logger.info("Agent finished - final answer generated")
        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}", exc_info=True)
    
    def finalize_tracking(self, final_answer: str) -> Dict[str, Any]:
        """Finalize both tracers and return combined results"""
        try:
            # End CoT trace
            cot_trace = self.cot_tracer.end_trace(final_decision=final_answer)
            
            # Finalize attribution report
            attribution_report = self.attribution_tracker.finalize_report(final_answer)
            
            return {
                "chain_of_thought": cot_trace.model_dump(),  # Fixed: using model_dump()
                "tool_attribution": attribution_report.model_dump(),  # Fixed: using model_dump()
                "summary": {
                    "total_reasoning_steps": len(cot_trace.reasoning_steps),
                    "reasoning_quality": cot_trace.reasoning_quality_score,
                    "tools_used": len(attribution_report.tools_used),
                    "unique_tools": attribution_report.tool_usage_summary.get("unique_tools_used", 0),
                    "total_execution_time": attribution_report.total_execution_time,
                    "answer_composition": attribution_report.answer_composition
                }
            }
        except Exception as e:
            logger.error(f"Error finalizing tracking: {e}", exc_info=True)
            return {
                "chain_of_thought": {},
                "tool_attribution": {},
                "summary": {},
                "error": str(e)
            }
    
    # Helper methods
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["calculate", "compute", "math", "sum", "multiply"]):
            return "calculation"
        elif any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            return "weather"
        elif any(word in query_lower for word in ["stock", "price", "market"]):
            return "financial"
        elif any(word in query_lower for word in ["news", "latest", "recent"]):
            return "news"
        elif any(word in query_lower for word in ["how", "what", "why", "when", "where", "who"]):
            return "factual"
        else:
            return "general"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        question_marks = query.count("?")
        
        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "moderate"
        else:
            return "complex"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query (simplified)"""
        words = query.split()
        entities = [word for word in words if word and len(word) > 0 and word[0].isupper() and len(word) > 2]
        return entities[:5]
    
    def _determine_intent(self, query: str) -> str:
        """Determine user intent"""
        query_lower = query.lower()
        
        if query_lower.startswith(("what", "who", "where", "when")):
            return "information_seeking"
        elif query_lower.startswith(("how",)):
            return "instruction_seeking"
        elif any(word in query_lower for word in ["compare", "versus", "vs"]):
            return "comparison"
        elif "?" in query:
            return "question"
        else:
            return "general_inquiry"
    
    def _extract_alternatives(self, reasoning: str) -> List[str]:
        """Extract alternative tools considered from reasoning"""
        common_tools = [
            "search", "calculator", "wikipedia", "weather",
            "stock", "document", "web"
        ]
        alternatives = [tool for tool in common_tools if tool in reasoning.lower()]
        return alternatives[:3]
    
    def _estimate_confidence(self, reasoning: str) -> float:
        """Estimate confidence from reasoning text"""
        confidence_indicators = {
            "clearly": 0.9,
            "definitely": 0.9,
            "probably": 0.7,
            "might": 0.5,
            "maybe": 0.5,
            "unsure": 0.3
        }
        
        reasoning_lower = reasoning.lower()
        for indicator, score in confidence_indicators.items():
            if indicator in reasoning_lower:
                return score
        
        return 0.7
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool by type"""
        tool_name_lower = tool_name.lower()
        
        if "search" in tool_name_lower or "web" in tool_name_lower:
            return "search"
        elif "calculator" in tool_name_lower or "math" in tool_name_lower:
            return "calculator"
        elif "document" in tool_name_lower or "retriev" in tool_name_lower:
            return "retriever"
        elif "weather" in tool_name_lower:
            return "weather_api"
        elif "stock" in tool_name_lower or "finance" in tool_name_lower:
            return "financial_api"
        elif "wikipedia" in tool_name_lower:
            return "encyclopedia"
        else:
            return "other"
    
    def _calculate_contribution_score(self, tool_name: str, output: str) -> float:
        """Calculate how much this tool contributed to the answer"""
        base_score = 0.5
        
        if output and len(output) > 100:
            base_score += 0.2
        
        critical_tools = ["search", "document", "retriever"]
        if any(ct in tool_name.lower() for ct in critical_tools):
            base_score += 0.2
        
        if "error" in output.lower() or "failed" in output.lower():
            base_score = 0.1
        
        return min(1.0, base_score)
    
    def _add_source_attribution(self, tool_name: str, output: str):
        """Add source attribution based on tool output"""
        try:
            if "document" in tool_name.lower():
                self.attribution_tracker.add_source_attribution(
                    source_type="internal_document",
                    source_identifier=tool_name,
                    information=output[:200],
                    relevance_score=0.8
                )
            elif "web" in tool_name.lower() or "search" in tool_name.lower():
                self.attribution_tracker.add_source_attribution(
                    source_type="web_search",
                    source_identifier=tool_name,
                    information=output[:200],
                    relevance_score=0.7
                )
            elif "wikipedia" in tool_name.lower():
                self.attribution_tracker.add_source_attribution(
                    source_type="encyclopedia",
                    source_identifier="Wikipedia",
                    information=output[:200],
                    relevance_score=0.9
                )
        except Exception as e:
            logger.error(f"Error adding source attribution: {e}")
    
    def _evaluate_tool_output(self, output: str) -> Dict[str, Any]:
        """Evaluate the quality of tool output"""
        if not output or len(output) < 10:
            return {
                "assessment": "insufficient",
                "quality_score": 0.2,
                "reasoning": "Output is too short or empty"
            }
        
        if "error" in output.lower() or "failed" in output.lower():
            return {
                "assessment": "error",
                "quality_score": 0.1,
                "reasoning": "Tool execution resulted in an error"
            }
        
        if len(output) > 200:
            return {
                "assessment": "comprehensive",
                "quality_score": 0.9,
                "reasoning": "Output contains substantial information"
            }
        elif len(output) > 50:
            return {
                "assessment": "adequate",
                "quality_score": 0.7,
                "reasoning": "Output contains useful information"
            }
        else:
            return {
                "assessment": "minimal",
                "quality_score": 0.5,
                "reasoning": "Output is brief but may be sufficient"
            }
    
    def _estimate_answer_confidence(self, final_answer: str) -> float:
        """Estimate confidence in the final answer"""
        confidence = 0.5
        
        if len(final_answer) > 100:
            confidence += 0.2
        
        if "." in final_answer and final_answer.count(".") > 1:
            confidence += 0.1
        
        uncertainty_words = ["might", "maybe", "possibly", "unclear", "unsure"]
        if any(word in final_answer.lower() for word in uncertainty_words):
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
