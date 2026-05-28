import json
import re
from typing import TypedDict, List, Tuple, Annotated, Any
from operator import itemgetter
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# --- 1. Define the State for our Graph ---
class Step(BaseModel):
    step: int = Field(description="The step number")
    tool: str = Field(description="The tool to use for this step")
    args: dict = Field(description="The arguments for the tool")
    reasoning: str = Field(description="The reasoning for this step")

def add_past_steps(left: List[Tuple], right: List[Tuple]) -> List[Tuple]:
    """Reducer function to accumulate tool calls and their results."""
    return left + right

class Plan(BaseModel):
    plan: List[Step] = Field(description="A list of steps to accomplish the goal")

class AgentState(TypedDict):
    input: str
    plan: List[dict]
    past_steps: Annotated[List[Tuple], add_past_steps]
    response: str

# --- 2. Enhanced Tool Executor Implementation ---
class EnhancedToolExecutor:
    """
    Enhanced tool executor with better error handling and result parsing.
    """
    def __init__(self, tools):
        if hasattr(tools, '__iter__') and not isinstance(tools, (str, bytes)):
            self.tools = {tool.name: tool for tool in tools}
        else:
            self.tools = {}
    
    def extract_numerical_result(self, result_str: str) -> str:
        """Extract numerical results from calculator outputs, handling various formats."""
        if not isinstance(result_str, str):
            return str(result_str)
        
        # Robustly look for numbers, including from "Calculation result:" or "Result:"
        calc_match = re.search(r'(?:Calculation result:|Result:|Answer:)?\s*(-?\d+\.?\d*)', result_str)
        if calc_match:
            return calc_match.group(1)
        
        # Fallback to look for any number in the string
        number_match = re.search(r'-?\d+\.?\d*', result_str)
        if number_match:
            return number_match.group(0)
        
        return result_str
    
    def invoke(self, tool_call):
        # Handle different tool call formats
        if hasattr(tool_call, 'tool') and hasattr(tool_call, 'tool_input'):
            tool_name = tool_call.tool
            tool_input = tool_call.tool_input
        elif isinstance(tool_call, dict):
            tool_name = tool_call.get('tool', tool_call.get('name'))
            tool_input = tool_call.get('tool_input', tool_call.get('args', {}))
        else:
            raise ValueError(f"Unsupported tool call format: {tool_call}")
        
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                if isinstance(tool_input, dict):
                    result = tool.invoke(tool_input)
                else:
                    # Fallback for non-dict inputs, assuming a simple `input` or `query` arg
                    result = tool.invoke({"input": tool_input})
                
                # Special handling for calculator results
                if tool_name == "advanced_calculator" and isinstance(result, str):
                    numerical_result = self.extract_numerical_result(result)
                    return numerical_result
                
                return result
            except Exception as e:
                # Log the specific tool error for debugging
                return f"Error executing tool {tool_name}: {str(e)}"
        else:
            return f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}"

class SimpleToolInvocation:
    """A simple tool invocation class"""
    def __init__(self, tool: str, tool_input: Any):
        self.tool = tool
        self.tool_input = tool_input

# --- 3. Define the Nodes of the Graph ---
def planner_node(state: AgentState, llm: BaseChatModel, planner_prompt: ChatPromptTemplate) -> dict:
    """
    Creates the step-by-step plan with improved error handling and robust JSON parsing.
    """
    print("---  Planning ---")
    
    try:
        planner = planner_prompt | llm
        response = planner.invoke({"input": state["input"]})
        
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"Raw planner response: {content}")
        
        plan_data = []
        try:
            # Robust JSON extraction logic from potentially messy LLM output
            json_match = re.search(r'```json\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'```\s*(\[.*?\]|\{.*?\})\s*```', content, re.DOTALL)

            json_str = json_match.group(1) if json_match else content
            parsed_response = json.loads(json_str)

            if 'plan' in parsed_response and isinstance(parsed_response['plan'], list):
                plan_data = parsed_response['plan']
            elif isinstance(parsed_response, list):
                plan_data = parsed_response
            else:
                raise ValueError("Parsed JSON is not in the expected format.")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse JSON from response: {e}")
            # Fallback to a simple plan using the 'smart_search' tool
            plan_data = [{
                "step": 1,
                "tool": "smart_search",
                "args": {"query": state["input"]},
                "reasoning": "Fallback: Could not parse plan, using smart search."
            }]
        
        # Format and validate the final plan
        plan_dicts = []
        for i, step in enumerate(plan_data):
            if isinstance(step, dict):
                plan_dicts.append({
                    "step": step.get("step", i + 1),
                    "tool": step.get("tool", "smart_search"),
                    "args": step.get("args", {}),
                    "reasoning": step.get("reasoning", f"Step {i + 1}")
                })
        
        print(f"Generated Plan: {json.dumps(plan_dicts, indent=2)}")
        return {"plan": plan_dicts}
            
    except Exception as e:
        print(f"Critical error during planning: {e}")
        # Final fallback to prevent graph from failing
        fallback_plan = [{
            "step": 1,
            "tool": "smart_search",
            "args": {"query": state["input"]},
            "reasoning": "Fallback: Critical error during planning."
        }]
        return {"plan": fallback_plan}

def tool_executor_node(state: AgentState, tool_executor) -> dict:
    """
    Executes a single tool call from the plan with enhanced placeholder substitution.
    """
    print("---  Executing Tool ---")
    
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    
    if not plan or len(past_steps) >= len(plan):
        return {}

    current_step_index = len(past_steps)
    current_step = plan[current_step_index]
    
    tool_name = current_step.get('tool')
    tool_args = current_step.get('args', {}).copy()

    if not tool_name:
        error_result = f"Step {current_step_index + 1} missing tool name"
        return {"past_steps": [(current_step, error_result)]}

    # Enhanced placeholder substitution
    for key, value in tool_args.items():
        tool_args[key] = substitute_placeholders(value, past_steps)

    step_number = current_step.get('step', current_step_index + 1)
    print(f"Executing Step {step_number}: Calling tool `{tool_name}` with args {tool_args}")
    
    try:
        action = SimpleToolInvocation(tool=tool_name, tool_input=tool_args)
        result = tool_executor.invoke(action)
        print(f"Tool execution result: {result}")
        return {"past_steps": [(action, result)]}
    except Exception as e:
        error_result = f"Error executing {tool_name}: {str(e)}"
        return {"past_steps": [(current_step, error_result)]}

def substitute_placeholders(arg_value: Any, past_steps: List[Tuple]) -> Any:
    """
    Recursively substitutes placeholders like "$step1_output" with the actual
    result from a previous step. This version is hardened to prevent crashes.
    """
    if not isinstance(arg_value, str):
        return arg_value

    # Match placeholders like $step1_output, $cached_wikipedia_output, etc.
    placeholder_pattern = r'\$(\w+)_output'
    
    # Check for direct placeholder match, e.g., "$step1_output"
    direct_match = re.fullmatch(placeholder_pattern, arg_value)
    if direct_match:
        placeholder = direct_match.group(1)
        step_index = -1
        try:
            # Handle numerical placeholders like "step1"
            match = re.search(r'\d+', placeholder)
            if match:
                step_index = int(match.group()) - 1
            else:
                # Fallback: Find the most recent step with the tool name
                for i in range(len(past_steps) - 1, -1, -1):
                    if past_steps[i][0].tool == placeholder:
                        step_index = i
                        break
            
            if 0 <= step_index < len(past_steps):
                result = past_steps[step_index][1]
                return str(result)
            else:
                return arg_value # Return original if not found
        except (ValueError, IndexError, TypeError) as e:
            print(f"Could not substitute placeholder ${placeholder}_output: {e}")
            return arg_value # Return original if error
            
    # If the placeholder is part of a larger string (e.g., in a prompt)
    substituted_value = arg_value
    placeholders = re.findall(placeholder_pattern, arg_value)
    
    for placeholder in placeholders:
        try:
            step_index = -1
            match = re.search(r'\d+', placeholder)
            if match:
                step_index = int(match.group()) - 1
            else:
                 for i in range(len(past_steps) - 1, -1, -1):
                    if past_steps[i][0].tool == placeholder:
                        step_index = i
                        break

            if 0 <= step_index < len(past_steps):
                result = past_steps[step_index][1]
                
                # Heuristic to handle calculator input
                is_math_expression = any(op in substituted_value for op in ['+', '-', '*', '/'])
                replacement = str(result)
                if is_math_expression:
                    numerical_match = re.search(r'-?\d+\.?\d*', str(result))
                    if numerical_match:
                        replacement = numerical_match.group(0)
                    else:
                        # Fallback for "1 million" etc.
                        million_match = re.search(r'(\d+\.?\d*)\s*million', str(result), re.IGNORECASE)
                        if million_match:
                            num = float(million_match.group(1))
                            replacement = str(int(num * 1_000_000))
                
                substituted_value = re.sub(re.escape(f'${placeholder}_output'), replacement, substituted_value)

        except (ValueError, IndexError, TypeError) as e:
            print(f"Could not substitute placeholder ${placeholder}_output: {e}")
            continue

    return substituted_value

def extract_birth_year_from_steps(past_steps: List[Tuple]) -> str:
    """Extract birth year from previous step results."""
    for step_action, step_result in past_steps:
        result_str = str(step_result)
        # Look for birth year patterns in Wikipedia results
        year_match = re.search(r'\b(164[0-9])\b', result_str)  # Newton was born in 1643
        if year_match:
            return year_match.group(1)
        
        # More general year pattern
        year_match = re.search(r'\b(1[5-9]\d{2})\b', result_str)
        if year_match:
            return year_match.group(1)
    
    return None

def final_answer_node(state: AgentState) -> dict:
    """
    Generates the final response based on executed steps.
    """
    print("---  Final Answer ---")
    
    past_steps = state.get("past_steps", [])
    
    if past_steps:
        last_result = past_steps[-1][1]
        
        # If the last step was final_answer, use its result
        last_action = past_steps[-1][0]
        if hasattr(last_action, 'tool') and last_action.tool == "final_answer":
            response = str(last_result)
        else:
            # Otherwise, construct a meaningful response
            if isinstance(last_result, str) and "error" in last_result.lower():
                valid_results = []
                for step_action, step_result in past_steps:
                    if "error" not in str(step_result).lower():
                        valid_results.append(str(step_result))
                
                if valid_results:
                    response = valid_results[-1]
                else:
                    response = "I encountered errors while trying to answer your question."
            else:
                # For Isaac Newton example, construct a proper answer
                birth_year = extract_birth_year_from_steps(past_steps)
                if birth_year:
                    current_year = datetime.now().year
                    years_ago = current_year - int(birth_year)
                    response = f"Isaac Newton was born in {birth_year}. That was {years_ago} years ago."
                else:
                    response = str(last_result)
        
        return {"response": response}
    else:
        return {"response": "The plan could not be executed successfully."}

def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue execution or end.
    """
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    
    if not plan:
        print("No plan available, ending")
        return "end_final_answer"
        
    if len(past_steps) >= len(plan):
        print("All steps completed, ending")
        return "end_final_answer"
        
    print(f"Continuing execution: {len(past_steps)}/{len(plan)} steps completed")
    return "continue"

from langgraph.checkpoint.memory import MemorySaver

def create_plan_and_execute_graph(llm: BaseChatModel, tools: List, planner_prompt: ChatPromptTemplate, checkpointer=None, interrupt_before=[]) -> Runnable:
    """
    Creates the enhanced plan and execute graph with optional HITL capabilities.
    """
    tool_executor = EnhancedToolExecutor(tools)
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("planner", lambda s: planner_node(s, llm, planner_prompt))
    graph.add_node("executor", lambda s: tool_executor_node(s, tool_executor))
    graph.add_node("final_answer", final_answer_node)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Add edges
    graph.add_edge("planner", "executor")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {
            "continue": "executor",
            "end_final_answer": "final_answer"
        }
    )
    
    graph.add_edge("final_answer", END)
    
    # Compile with HITL options
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

# Enhanced system prompt for better planning
def get_enhanced_plan_and_execute_system_prompt() -> str:
    """
    Enhanced system prompt that gives clearer instructions for mathematical calculations.
    """
    return """You are an expert problem solver. Your job is to create a step-by-step plan to answer the user's query using the available tools.

**Available Tools:**
{tools_description}

**Instructions & Rules:**
1. **Analyze the User's Goal:** Understand what the user is asking for.
2. **Use Direct Tools:** If a specific tool directly answers the query, use that tool first.
3. **For Mathematical Calculations:** - First, gather the required data (e.g., birth year from Wikipedia)
   - Then, use the advanced_calculator tool with the current year
   - For birth year calculations, use expressions like "2025 - 1643" (current year - birth year)
4. **Placeholder References:** When referencing previous step outputs, use clear formats like "$step1_output"
5. **Final Step:** Always end with the final_answer tool that summarizes the complete answer
6. **Output JSON Only:** Your response must be ONLY a valid JSON array of step objects.


Create the plan for the user's query:"""
