"""
Core Agent Module
Contains ReasoningAgentSystem and ReasoningEngine.
"""
import logging
import time
import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field

# Updated imports to point to new modules
from src.configuration import AgentConfig, setup_logging
from src.model import ChatOllama
from src.reasoning_agent.math_engine import SymbolicMathEngine
from src.reasoning_agent.physics_solver import PhysicsSolver2D
from src.reasoning_agent.knowledge_manager import KnowledgeManager, KnowledgeDomain
from src.reasoning_agent.parsers import LLMProblemParser, ProblemDecomposer
from src.reasoning_agent.utils import UnitManager, ErrorRecoverySystem, ConfidenceEstimator, AnswerVerifier

# Constants
SYMPY_AVAILABLE = True # Assuming true as we check in math_engine
OLLAMA_AVAILABLE = True # Controlled by config usually

@dataclass
class ReasoningStep:
    step_num: int
    thought: str
    action: str
    observation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

class OllamaClient:
    """Wrapper for Ollama interaction"""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger("reasoning_agent")
        self.model = None
        self._init_model()
    
    def _init_model(self):
        try:
            # Check if ChatOllama is available/working? 
            # This logic mimics original file
            self.model = ChatOllama(
                model=self.config.ollama_model,
                temperature=self.config.temperature,
                num_ctx=self.config.num_ctx,
                num_predict=self.config.num_predict,
                base_url=self.config.ollama_base_url
            )
            self.logger.info(f"Ollama '{self.config.ollama_model}' ready")
        except Exception as e:
            self.logger.error(f"Ollama init failed: {e}")
    
    def generate(self, prompt: str, system: str = None) -> str:
        if not self.model:
            return "[LLM unavailable - ensure Ollama is running]"
        try:
            msgs = []
            if system:
                msgs.append(("system", system))
            msgs.append(("human", prompt))
            return self.model.invoke(msgs).content
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"[LLM error: {e}]"

class ReasoningEngine:
    """Core reasoning engine with dynamic capabilities"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger("reasoning_agent")
        
        # Initialize Sub-components
        self.knowledge = KnowledgeManager(data_path=os.path.join(os.getcwd(), "data", "knowledge_base"))
        self.llm = OllamaClient(config)
        self.math_engine = SymbolicMathEngine()
        self.unit_manager = UnitManager(self.knowledge)
        self.verifier = AnswerVerifier(self.math_engine)
        self.confidence_estimator = ConfidenceEstimator()
        self.decomposer = ProblemDecomposer(self)
        self.parser = LLMProblemParser(self.llm) if config.enable_llm_parsing else None
        self.error_recovery = ErrorRecoverySystem(self)

    def solve(self, query: str, show_work: bool = True) -> Dict[str, Any]:
        """Main solving entry point"""
        self.logger.info(f"{'='*60}\\nSOLVING: {query}\\n{'='*60}")
        
        # CHECK FOR DECOMPOSITION FIRST
        decomposed = self.decomposer.decompose_and_solve(query)
        if decomposed:
            conf = self.confidence_estimator.estimate_confidence(
                "decomposed", {"parse_success": True}, {"error": None}, None
            )
            return {
                "success": True,
                "query": query,
                "decomposed": True,
                "answer": decomposed["answer"],
                "sub_problems": decomposed.get("sub_problems", []),
                "method": decomposed.get("method", "decomposition"),
                "confidence": conf,
                "steps": [],
                "details": decomposed
            }
       
        # Continue with normal solving...
        steps = []
        
        # Step 1: Parse problem
        parsed = self._parse_problem(query, steps)
        
        # Step 2: Route to appropriate solver
        problem_type = parsed.get('type', 'qualitative')
        result = self._route_problem(query, parsed, problem_type, steps)
        
        # Step 3: Confidence Estimation
        verification = result.get("details", {}).get("verification")
        confidence = self.confidence_estimator.estimate_confidence(
            problem_type, parsed, result.get("details", {}), verification
        )
        
        return {
            "success": True,
            "query": query,
            "parsed": parsed,
            "problem_type": problem_type,
            "answer": result["answer"],
            "confidence": confidence,
            "steps": [s.to_dict() for s in steps] if show_work else [],
            "details": result.get("details", {})
        }

    def _parse_problem(self, query: str, steps: List[ReasoningStep]) -> Dict:
        """Parse the problem with error recovery"""
        if self.parser and self.config.enable_llm_parsing:
            clean_query = re.sub(r'^\[Subject:\s*\w+\]\s*', '', query, flags=re.I)
            parsed = self.parser.parse(clean_query)
            parse_method = "LLM-guided" if parsed.get('parse_success') else "fallback"

            if not parsed.get('parse_success'):
                recovery = self.error_recovery.attempt_recovery('parsing', {'problem': clean_query})
                if recovery.get('recovered'):
                    parsed = recovery['parsed']
                    parse_method = f"recovered ({recovery['strategy']})"
                    self.logger.info(f"✓ Parsing recovered via {recovery['strategy']}")
        else:
            clean_query = re.sub(r'^\[Subject:\s*\w+\]\s*', '', query, flags=re.I)
            # Use fallback parser if LLM parser disabled or not init
            parsed = {
                'type': 'qualitative',
                'parse_success': True,
                'problem': clean_query
            }
            parse_method = "simple"
        
        steps.append(ReasoningStep(
            len(steps)+1,
            f"Parsed problem using {parse_method} method",
            "parse",
            f"Type: {parsed.get('type', 'unknown')}"
        ))
        return parsed
    
    def _route_problem(self, query: str, parsed: Dict, problem_type: str, steps: List[ReasoningStep]) -> Dict:
        """Route problem to appropriate solver"""
        try:
            domain = KnowledgeDomain(parsed.get('domain', 'general'))
            
            if problem_type == 'math':
                return self._solve_math_problem(parsed, steps)
            elif problem_type == 'physics':
                return self._solve_physics(parsed, steps)
            else:
                return self._solve_qualitative(query, parsed, domain, steps)
        except Exception as e:
            return self._handle_solver_error(e, query, parsed, steps)
    
    def _solve_math_problem(self, parsed: Dict, steps: List[ReasoningStep]) -> Dict:
        """Solve mathematical problems"""
        subtype = parsed.get('subtype', 'general')
        var = parsed.get('variable', 'x')
        expr = parsed.get('expression', '')
        
        # Retry logic for symbolic computation
        attempt = 0
        max_attempts = 3
        result = None
        
        while attempt < max_attempts and not result:
            try:
                if subtype == 'derivative':
                    result = self.math_engine.compute_derivative(expr, var)
                elif subtype == 'integral':
                    result = self.math_engine.compute_integral(expr, var)
                elif subtype == 'equation':
                    result = self.math_engine.solve_equation(expr, var)
                else:
                    result = self.math_engine.simplify_expression(expr)
                
                # Verify if result is valid
                if result and 'error' in result:
                    self.logger.warning(f"Attempt {attempt+1} failed")
                    result = None
                    attempt += 1
                else:
                    break
            except Exception as e:
                self.logger.error(f"Math op error: {e}")
                attempt += 1
        
        # Format answer
        if result and 'error' not in result:
            answer = self._format_math_answer(subtype, result, var)
        else:
            answer = f"Error computing {subtype}: {result.get('error') if result else 'Failed'}"
            
        steps.append(ReasoningStep(3, "Applied symbolic computation", "sympy_calculation", f"Operation: {subtype}"))
        return {"answer": answer, "details": result if result else {}}

    def _format_math_answer(self, subtype: str, result: Dict, var: str) -> str:
        """Format answer string"""
        # (Simplified version of original formatting logic)
        if subtype == 'derivative':
            return f"Derivative: {result.get('derivative')}"
        elif subtype == 'integral':
            return f"Integral: {result.get('result')}"
        elif subtype == 'equation':
            return f"Solutions: {result.get('solutions')}"
        return f"Result: {result}"

    def _solve_physics(self, parsed: Dict, steps: List[ReasoningStep]) -> Dict:
        """Solve physics problems"""
        subtype = parsed.get('subtype', 'kinematics')
        known = parsed.get('known_values', {})
        
        # Standardize units
        if self.unit_manager.available:
            unit_result = self.unit_manager.standardize_physics_units(known)
            known = unit_result["values"]
            if unit_result["conversions"]:
                steps.append(ReasoningStep(
                    len(steps)+1, "Standardized units to SI",
                    "unit_conversion", "; ".join(unit_result["conversions"])
                ))
        
        # Route to physics solver
        solver = PhysicsSolver2D(self.knowledge)
        if subtype == 'projectile':
            result = solver.solve_projectile_motion(known)
            answer = str(result.get('results', result))
        elif subtype == 'forces' and 'angle' in known: # Incline check
            result = solver.solve_incline_with_friction(known)
            answer = str(result.get('results', result))
        else:
            # Use appropriate kinematics solver
            find = parsed.get('find', 'd')
            result = solver.solve_kinematics_1d(known, find)
            answer = f"Result: {result.get('results', {}).get(find, result.get('error'))}"
            
        steps.append(ReasoningStep(3, "Computed physics result", "physics_calculation", str(result)))
        return {"answer": answer, "details": result}

    def _solve_qualitative(self, query: str, parsed: Dict, domain: KnowledgeDomain, steps: List[ReasoningStep]) -> Dict:
        """Solve qualitative problems using knowledge base"""
        steps.append(ReasoningStep(len(steps)+1, f"Qualitative problem in {domain.value}", "knowledge_retrieval", "Retrieving concepts"))
        
        kb_results = self.knowledge.retrieve(query, domain, k=3)
        context = "\\n".join([f"**{k['metadata']['name']}**: {k['page_content']}" for k in kb_results])
        
        prompt = f"Question: {query}\\n\\nKnowledge:\\n{context}\\n\\nAnswer:"
        answer = self.llm.generate(prompt)
        
        return {"answer": answer, "details": {"source": "knowledge_base", "concepts": kb_results}}

    def _handle_solver_error(self, error: Exception, query: str, parsed: Dict, steps: List) -> Dict:
        """Handle errors with knowledge retrieval fallback"""
        self.logger.error(f"Solver failed: {error}")
        
        # Fallback to knowledge base
        kb_results = self.knowledge.retrieve(query, k=3)
        answer = f"**Error:** {str(error)}\\n\\n**Related Knowledge:**\\n"
        for k in kb_results:
            answer += f"- {k['metadata']['name']}: {k['page_content'][:100]}...\\n"
            
        return {"answer": answer, "details": {"error": str(error), "fallback": "knowledge"}}

class ReasoningAgentSystem:
    """Main Agent System Entry Point"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        # Initialize logging if not already done?
        # setup_logging(self.config) # Assuming this handles it
        self.engine = ReasoningEngine(self.config)
        
    def solve(self, problem: str, show_work: bool = True) -> Dict[str, Any]:
        if not problem or not problem.strip():
            return {"success": False, "error": "Empty problem"}
        try:
            return self.engine.solve(problem.strip(), show_work)
        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Simple test
    agent = ReasoningAgentSystem()
    print(agent.solve("Derivative of x^2"))
