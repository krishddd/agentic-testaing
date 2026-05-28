"""
Parsers Module
Contains LLMProblemParser and ProblemDecomposer.
"""
import logging
import re
import json
from typing import Dict, Any, List

logger = logging.getLogger("reasoning_agent")

class LLMProblemParser:
    PARSING_PROMPT = '''You are a precise problem parser. Extract structured information from the problem.
Respond ONLY with valid JSON enclosed in <json> tags. No other text before or after.

For MATH problems, extract:
{"type": "math", "subtype": "equation|derivative|integral|simplify|system|limit|optimization", 
 "expression": "the mathematical expression", "variable": "x", 
 "operation": "solve|differentiate|integrate|simplify|evaluate|maximize|minimize",
 "bounds": {"lower": num_or_"infinity", "upper": num_or_"infinity"} or null,
 "constraints": [] or null}

CRITICAL FOR INTEGRALS:
- If you see "from A to B", extract bounds as {"lower": A, "upper": B}
- For infinity: use the STRING "infinity" (not null, not a number)
- For negative infinity: use the STRING "-infinity"
- Convert e^(...) to exp(...) in expressions
- Examples: 
  * "from 0 to infinity" → {"lower": 0, "upper": "infinity"}
  * "from -infinity to 0" → {"lower": "-infinity", "upper": 0}
  * "e^(-x)" → "exp(-x)"

For PHYSICS problems, extract:
{"type": "physics", "subtype": "kinematics|forces|energy|momentum|circuits|waves|projectile|rotation",
 "known_values": {"variable_name": "value_with_unit", ...},
 "find": "variable_to_solve_for",
 "scenario": "brief description"}

For QUALITATIVE problems:
{"type": "qualitative", "domain": "economics|political_science|psychology|sociology|statistics",
 "topic": "specific topic", "question_type": "explain|compare|analyze|predict|define"}

Examples:

Problem: "Find the derivative of sin(x²) with respect to x"
<json>{"type": "math", "subtype": "derivative", "expression": "sin(x**2)", "variable": "x", "operation": "differentiate"}</json>

Problem: "Calculate ∫x^2 dx from -1 to 1"
<json>{"type": "math", "subtype": "integral", "expression": "x**2", "variable": "x", "operation": "integrate", "bounds": {"lower": -1, "upper": 1}}</json>

Problem: "A 10kg box slides down a 30° incline with friction coefficient 0.2. Find acceleration."
<json>{"type": "physics", "subtype": "forces", "known_values": {"m": "10 kg", "angle": "30 deg", "mu": "0.2"}, "find": "a", "scenario": "incline with friction"}</json>

Now parse this problem:
'''
    def __init__(self, llm_client):
        """Initialize parser with LLM client"""
        self.llm = llm_client
        self.logger = logger
    
    def classify_query_intent(self, problem: str) -> str:
        """
        Classify query intent before parsing to distinguish definitional from computational queries.
        Returns: 'definitional', 'conceptual', 'computational', 'procedural', 'ambiguous'
        """
        problem_lower = problem.lower().strip()
        
        # Definitional patterns - seeking definitions or explanations of concepts
        definitional_patterns = [
            r'^(?:what|define|meaning of|definition of)\s+(?:is|are|does|means)',
            r'^(?:numbers|equations|functions|formulas|concepts)\s+of\s+the\s+form',
            r'^(?:a|an)\s+\w+\s+is\s+(?:a|an|the)',
            r'explain\s+(?:what|the)\s+(?:is|are)',
            r'^(?:tell me|show me)\s+(?:about|what)',
        ]
        
        # Conceptual patterns - asking about theories, principles, laws
        conceptual_patterns = [
            r'^(?:explain|describe|discuss)\s+(?:the|how|why)',
            r'^(?:why|how does)\s+',
            r'(?:concept|theory|principle|law)\s+of',
            r'state\s+the\s+(?:law|theorem|principle)',
        ]
        
        # Computational patterns (strong indicators) - asking to calculate/solve
        computational_patterns = [
            r'^(?:solve|calculate|compute|find|evaluate|determine)\s+',
            r'\w+\s*=\s*\d',  # Contains equation like x=5
            r'^(?:integrate|differentiate|derive)\s+',
            r'simplify\s+',
            r'factor\s+',
        ]
        
        # Procedural patterns - asking how to do something
        procedural_patterns = [
            r'^how\s+(?:do|to|can)\s+(?:i|you|we)',
            r'steps?\s+to\s+',
            r'method\s+(?:for|to)\s+',
            r'procedure\s+for',
        ]
        
        # Check in priority order (computational first to avoid false positives)
        if any(re.search(p, problem_lower) for p in computational_patterns):
            return 'computational'
        if any(re.search(p, problem_lower) for p in definitional_patterns):
            return 'definitional'
        if any(re.search(p, problem_lower) for p in conceptual_patterns):
            return 'conceptual'
        if any(re.search(p, problem_lower) for p in procedural_patterns):
            return 'procedural'
        
        # Additional heuristic: short queries without verbs are likely definitional
        # e.g., "Complex numbers", "Pythagorean theorem"
        word_count = len(problem_lower.split())
        has_action_verb = any(v in problem_lower for v in ['solve', 'find', 'calculate', 'compute', 'evaluate'])
        has_equation = '=' in problem or re.search(r'\w+\^\d', problem)
        
        if word_count <= 5 and not has_action_verb and not has_equation:
            return 'definitional'
        
        return 'ambiguous'   
        
    def parse(self, problem: str) -> Dict[str, Any]:
        """Parse problem using LLM with better JSON extraction"""
        
        # NEW: Classify query intent first to avoid treating definitions as equations
        intent = self.classify_query_intent(problem)
        
        # If definitional/conceptual, route to knowledge retrieval instead of computation
        if intent in ['definitional', 'conceptual']:
            self.logger.info(f"✓ Detected {intent} query: '{problem[:50]}...'")
            return {
                'type': 'qualitative',
                'intent': intent,
                'query': problem,
                'parse_success': True,
                'raw_problem': problem
            }
        
        # Continue with LLM parsing for computational queries
        prompt = self.PARSING_PROMPT + f'"{problem}"'
        
        try:
            response = self.llm.generate(prompt)
            
            # Try extracting JSON from <json> tags first
            json_match = re.search(r'<json>\s*(\{.*?\})\s*</json>', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fallback: try removing markdown code blocks
                response = response.strip()
                response = re.sub(r'```(?:json)?\s*', '', response)
                response = response.replace('```', '')
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            parsed = json.loads(json_str)
            parsed['raw_problem'] = problem
            parsed['parse_success'] = True
            self.logger.info(f"✓ LLM parsing successful: {parsed.get('type')}/{parsed.get('subtype')}")
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"LLM parse failed: {e}, using fallback")
            return self._fallback_parse(problem)
        except Exception as e:
            self.logger.warning(f"Parse error: {e}")
            return self._fallback_parse(problem)
            
    def _fallback_parse(self, problem: str) -> Dict[str, Any]:
        """Enhanced regex-based fallback parsing with intent awareness"""
        
        # NEW: Check intent first before regex parsing
        intent = self.classify_query_intent(problem)
        if intent in ['definitional', 'conceptual']:
            self.logger.info(f"✓ Fallback parser detected {intent} query")
            return {
                'type': 'qualitative',
                'intent': intent,
                'query': problem,
                'parse_success': False,
                'raw_problem': problem
            }
        
        problem_lower = problem.lower()
        result = {"raw_problem": problem, "parse_success": False}
        
        # CRITICAL: Strip subject hint prefix if present
        problem_clean = re.sub(r'^\[Subject:\s*\w+\]\s*', '', problem, flags=re.I)
        problem_lower_clean = problem_clean.lower()
        
        # Math detection - EXPANDED PATTERNS
        derivative_keywords = ['derivative', 'differentiate', "d/dx", "f'(", 'rate of change', 
                            'slope', 'tangent', 'dy/dx']
        integral_keywords = ['integral', 'integrate', '∫', 'antiderivative', 'area under']
        optimization_keywords = ['maximize', 'minimize', 'maximum', 'minimum', 'optimal', 
                                'max', 'min', 'largest', 'smallest']
        
        # ========================================================================
        # PRIORITY 1: EQUATIONS (Check FIRST to prevent qualitative classification)
        # ========================================================================
        equation_patterns = [
            r'solve\s+[^=]+=\s*[^=]+',  # "solve x^2 = 4"
            r'[a-z]\^?\d.*?=\s*\d',      # "x^2 = 0" or "x² = 0"
            r'\w+\s*=\s*0',               # "x² - 5x + 6 = 0"
            r'equation',                  # explicit "equation" keyword
            r'quadratic|cubic|polynomial', # equation types
        ]
        
        if any(re.search(pattern, problem_lower_clean) for pattern in equation_patterns):
            result.update({"type": "math", "subtype": "equation", "operation": "solve"})
            # Try to extract equation
            eq_match = re.search(r'solve\s+(.+?)(?:\s+for|$)', problem_clean, re.I) or \
                    re.search(r'([^=]+=.+?)(?:\s+for|$)', problem_clean)
            result["expression"] = eq_match.group(1).strip() if eq_match else problem_clean
            result["variable"] = "x"
            return result  # Early return
        
        # ========================================================================
        # PRIORITY 2: DERIVATIVES
        # ========================================================================
        if any(kw in problem_lower_clean for kw in derivative_keywords):
            expr = None
            for pattern in [
                r'(?:of|for)\s+(.+?)(?:\s+with respect to|\s+at|$)',
                r'rate of change of\s+(.+?)(?:\s|$)',
                r"f'?\(x\)\s*=\s*(.+?)(?:\s|$)",
                r'derivative\s+(.+?)(?:\s+with|\s+at|$)'
            ]:
                match = re.search(pattern, problem_clean, re.I)
                if match:
                    expr = match.group(1).strip()
                    break
            
            result.update({
                "type": "math", "subtype": "derivative", "operation": "differentiate",
                "expression": expr if expr else problem_clean,
                "variable": "x"
            })
            return result
        
        # ========================================================================
        # PRIORITY 3: OPTIMIZATION
        # ========================================================================
        elif any(kw in problem_lower_clean for kw in optimization_keywords):
            expr_match = re.search(r'(?:f\(x\)\s*=\s*|maximize|minimize)\s*(.+?)(?:\s+subject|$)', problem_clean, re.I)
            result.update({
                "type": "math", "subtype": "optimization",
                "operation": "maximize" if any(k in problem_lower_clean for k in ['maximize', 'maximum', 'max', 'largest']) else "minimize",
                "expression": expr_match.group(1).strip() if expr_match else problem_clean,
                "variable": "x"
            })
            return result
        
        # ========================================================================
        # PRIORITY 4: INTEGRALS - ENHANCED BOUNDS EXTRACTION
        # ========================================================================
        elif any(kw in problem_lower_clean for kw in integral_keywords):
            result.update({"type": "math", "subtype": "integral", "operation": "integrate"})
            
            # Extract expression - multiple patterns
            expr = None
            for pattern in [
                r'(?:integral of|integrate)\s+(.+?)(?:\s+from|\s+with|$)',
                r'∫\s*(.+?)\s*d[xyz]',
                r'integrate\s+(.+?)(?:\s+from|$)',
                r'(?:of|for)\s+(.+?)(?:\s+from|\s+with|$)'
            ]:
                match = re.search(pattern, problem_clean, re.I)
                if match:
                    expr = match.group(1).strip()
                    break
            
            if not expr:
                # Fallback: extract anything that looks like a function
                expr_match = re.search(r'([a-z^()\-+*/\d\s]+)(?:\s+from|$)', problem_clean, re.I)
                expr = expr_match.group(1).strip() if expr_match else problem_clean
            
            result["expression"] = expr
            result["variable"] = "x"
            
            # ===== CRITICAL: ENHANCED BOUNDS EXTRACTION =====
            bounds = None
            
            # Pattern 1: "from X to Y" - handle numbers and infinity
            bounds_match = re.search(
                r'from\s+([-\d.]+|infinity|inf|∞|−∞|-infinity)\s+to\s+([-\d.]+|infinity|inf|∞|−∞|-infinity)',
                problem_lower_clean,
                re.I
            )
            
            if bounds_match:
                lower_str = bounds_match.group(1).strip()
                upper_str = bounds_match.group(2).strip()
                
                # Convert to proper types
                def parse_bound(bound_str):
                    bound_lower = bound_str.lower()
                    if bound_lower in ['infinity', 'inf', '∞']:
                        return 'infinity'
                    elif bound_lower in ['-infinity', '-inf', '−∞', '-∞']:
                        return '-infinity'
                    else:
                        try:
                            return float(bound_str)
                        except:
                            return bound_str
                
                lower_val = parse_bound(lower_str)
                upper_val = parse_bound(upper_str)
                
                bounds = {"lower": lower_val, "upper": upper_val}
                result["bounds"] = bounds
            
            # Pattern 2: "between X and Y"
            elif re.search(r'between\s+([-\d.]+)\s+and\s+([-\d.]+)', problem_lower_clean):
                bounds_match = re.search(r'between\s+([-\d.]+)\s+and\s+([-\d.]+)', problem_clean)
                if bounds_match:
                    result["bounds"] = {
                        "lower": float(bounds_match.group(1)),
                        "upper": float(bounds_match.group(2))
                    }
            
            # Pattern 3: "[X, Y]" notation
            elif re.search(r'\[(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)\]', problem_clean):
                bounds_match = re.search(r'\[(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)\]', problem_clean)
                if bounds_match:
                    result["bounds"] = {
                        "lower": float(bounds_match.group(1)),
                        "upper": float(bounds_match.group(2))
                    }
            
            return result
        
        # ========================================================================
        # PRIORITY 5: PHYSICS
        # ========================================================================
        projectile_keywords = ['thrown', 'launched', 'projectile', 'trajectory', 'angle']
        incline_keywords = ['incline', 'ramp', 'slope', 'friction']
        
        if any(kw in problem_lower_clean for kw in projectile_keywords):
            result.update({
                "type": "physics", "subtype": "projectile",
                "known_values": self._extract_physics_values(problem_clean),
                "scenario": "projectile motion"
            })
            find_match = re.search(r'(?:find|calculate|what is)\s+(?:the\s+)?(\w+)', problem_clean, re.I)
            result["find"] = find_match.group(1) if find_match else "range"
            return result
        
        elif any(kw in problem_lower_clean for kw in incline_keywords):
            result.update({
                "type": "physics", "subtype": "forces",
                "known_values": self._extract_physics_values(problem_clean),
                "scenario": "incline with friction"
            })
            find_match = re.search(r'(?:find|calculate)\s+(?:the\s+)?(\w+)', problem_clean, re.I)
            result["find"] = find_match.group(1) if find_match else "a"
            return result
        
        elif any(kw in problem_lower_clean for kw in ['velocity', 'acceleration', 'displacement', 'force', 'mass', 'm/s']):
            is_kinematics = any(k in problem_lower_clean for k in ['velocity', 'displacement', 'time'])
            result.update({
                "type": "physics",
                "subtype": "kinematics" if is_kinematics else "forces",
                "known_values": self._extract_physics_values(problem_clean)
            })
            find_match = re.search(r'(?:find|calculate|what is|determine)\s+(?:the\s+)?(\w+)', problem_clean, re.I)
            result["find"] = find_match.group(1) if find_match else None
            return result
        
        # ========================================================================
        # PRIORITY 6: QUALITATIVE (Last Resort)
        # ========================================================================
        domain = "general"
        if any(kw in problem_lower_clean for kw in ['supply', 'demand', 'price', 'market', 'gdp', 'inflation', 'elasticity']):
            domain = "economics"
        elif any(kw in problem_lower_clean for kw in ['democracy', 'autocracy', 'political', 'government', 'regime', 'election']):
            domain = "political_science"
        elif any(kw in problem_lower_clean for kw in ['memory', 'cognitive', 'behavior', 'psychology', 'learning', 'perception']):
            domain = "psychology"
        elif any(kw in problem_lower_clean for kw in ['mean', 'median', 'variance', 'probability', 'distribution', 'hypothesis', 'regression']):
            domain = "statistics"
        
        result.update({"type": "qualitative", "domain": domain})
        
        return result
    
    def _extract_physics_values(self, problem: str) -> Dict[str, float]:
        """Extract numeric values with units from physics problems"""
        # Also strip the subject hint prefix here
        problem = re.sub(r'^\[Subject:\s*\w+\]\s*', '', problem, flags=re.I)
        
        values = {}
        patterns = [
            (r'(?:initial\s+)?velocity\s*[=:]\s*([-+]?\d+\.?\d*)', 'v_i'),
            (r'final\s+velocity\s*[=:]\s*([-+]?\d+\.?\d*)', 'v_f'),
            (r'acceleration\s*[=:]\s*([-+]?\d+\.?\d*)', 'a'),
            (r'time\s*[=:]\s*([-+]?\d+\.?\d*)', 't'),
            (r'displacement\s*[=:]\s*([-+]?\d+\.?\d*)', 'd'),
            (r'distance\s*[=:]\s*([-+]?\d+\.?\d*)', 'd'),
            (r'mass\s*[=:]\s*([-+]?\d+\.?\d*)', 'm'),
            (r'force\s*[=:]\s*([-+]?\d+\.?\d*)', 'F'),
            (r'(\d+\.?\d*)\s*kg', 'm'),
            (r'(\d+\.?\d*)\s*m/s²', 'a'),
            (r'(\d+\.?\d*)\s*m/s(?!\^|²)', 'v_i'),
            (r'(\d+\.?\d*)\s*(?:seconds?|s\b)', 't'),
            (r'(\d+\.?\d*)\s*(?:meters?|m\b)(?!/)', 'd'),
            (r'(\d+\.?\d*)\s*N\b', 'F'),
        ]
        
        for pattern, var in patterns:
            match = re.search(pattern, problem, re.I)
            if match and var not in values:
                try:
                    values[var] = float(match.group(1))
                except:
                    continue
        
        return values

class ProblemDecomposer:
    """Break complex problems into sub-problems"""
    
    def __init__(self, reasoning_engine):
        self.engine = reasoning_engine
        self.logger = logger
    
    def decompose_and_solve(self, query: str) -> Dict[str, Any]:
        """Identify if problem needs decomposition and solve recursively"""
        
        # Detect patterns that require multi-step solutions
        query_lower = query.lower()
        
        # Pattern 1: Average velocity from acceleration function
        if re.search(r'average.*velocity.*a\(t\)|acceleration.*function', query_lower):
            return self._solve_average_velocity_from_acceleration(query)
        
        # Pattern 2: Optimization with constraints
        if 'subject to' in query_lower or 'constraint' in query_lower:
            return self._solve_constrained_optimization(query)
        
        # Pattern 3: "Find X, then find Y using X"
        if re.search(r'then|and then|next', query_lower):
            return self._solve_sequential_problem(query)
        
        # Pattern 4: Word problems requiring setup
        if len(query.split()) > 20 and any(kw in query_lower for kw in ['if', 'when', 'after']):
            return self._solve_word_problem(query)
        
        # No decomposition needed
        return None
    
    def _solve_average_velocity_from_acceleration(self, query: str) -> Dict:
        """
        Example: "Find average velocity from t=0 to t=5 if a(t) = 2t"
        Steps:
        1. Integrate a(t) to get v(t)
        2. Evaluate v(0) and v(5)
        3. Calculate average: (v(0) + v(5))/2
        """
        sub_problems = []
        
        # Extract acceleration function
        match = re.search(r'a\(t\)\s*=\s*(.+?)(?:\s|$)', query, re.I)
        if not match:
            return {"error": "Could not extract acceleration function"}
        
        a_t = match.group(1).strip()
        
        # Extract time bounds
        bounds = re.findall(r't\s*=\s*(\d+).*?t\s*=\s*(\d+)', query)
        if not bounds:
            return {"error": "Could not extract time bounds"}
        t0, t1 = float(bounds[0][0]), float(bounds[0][1])
        
        # Step 1: Integrate to get v(t)
        sub_problems.append({
            "step": 1,
            "description": f"Integrate a(t) = {a_t} to get v(t)",
            "problem": f"Integrate {a_t} with respect to t"
        })
        
        result1 = self.engine.solve(f"Integrate {a_t}", show_work=False)
        if not result1['success']:
            return {"error": "Failed to integrate acceleration", "details": result1}
        
        # Extract v(t) from result
        # This is simplified - in practice you'd parse the result more carefully
        v_t = result1.get('details', {}).get('result', 'unknown')
        
        sub_problems.append({
            "step": 2,
            "description": f"Got v(t) = {v_t}",
            "result": v_t
        })
        
        # Step 2: Evaluate at bounds (simplified - would use symbolic evaluation)
        answer = f"""**Multi-step Problem: Average Velocity from Acceleration**
**Sub-problems solved:**
1. Integrated a(t) = {a_t} to get v(t)
   → v(t) = {v_t}
2. To find average velocity from t={t0} to t={t1}:
   - Evaluate v({t0}) and v({t1})
   - Average = [v({t0}) + v({t1})]/2
**Decomposition:** This problem required:
→ Integration (calculus)
→ Function evaluation
→ Arithmetic mean calculation
See individual sub-problems above for details."""
        
        return {
            "decomposed": True,
            "sub_problems": sub_problems,
            "answer": answer,
            "method": "multi_step_decomposition"
        }
    
    def _solve_sequential_problem(self, query: str) -> Dict:
        """Handle problems with explicit sequential steps"""
        # Split by sequential indicators
        parts = re.split(r'\bthen\b|\band then\b|\bnext\b', query, flags=re.I)
        
        if len(parts) < 2:
            return None
        
        sub_problems = []
        
        for i, part in enumerate(parts, 1):
            part = part.strip()
            if not part:
                continue
            
            # Solve this sub-problem
            result = self.engine.solve(part, show_work=False)
            
            sub_problems.append({
                "step": i,
                "description": part,
                "result": result.get('answer', 'Failed'),
                "success": result['success']
            })
        
        final_answer = f"""**Sequential Problem Solution**
**Problem broken into {len(sub_problems)} steps:**
"""
        for sp in sub_problems:
            final_answer += f"\n**Step {sp['step']}:** {sp['description']}\n"
            final_answer += f"{'✓' if sp['success'] else '✗'} Result: {sp['result']}\n"
        
        return {
            "decomposed": True,
            "sub_problems": sub_problems,
            "answer": final_answer,
            "method": "sequential_decomposition"
        }
    
    def _solve_word_problem(self, query: str) -> Dict:
        """Attempt to decompose complex word problems"""
        # Use LLM to break down word problem
        if not hasattr(self.engine, 'llm') or not self.engine.llm.model:
            return None
        
        decomposition_prompt = f"""Break this problem into clear sequential steps:
Problem: {query}
Provide a numbered list of sub-problems to solve, in order. Each should be a simple, atomic task.
Format as:
1. [First sub-problem]
2. [Second sub-problem]
...
Only list the sub-problems, nothing else."""
        
        try:
            response = self.engine.llm.generate(decomposition_prompt)
            
            # Extract numbered steps
            steps = re.findall(r'^\d+\.\s*(.+?)$', response, re.MULTILINE)
            
            if len(steps) < 2:
                return None
            
            sub_problems = []
            for i, step in enumerate(steps, 1):
                result = self.engine.solve(step, show_work=False)
                sub_problems.append({
                    "step": i,
                    "description": step,
                    "result": result.get('answer', 'Failed')[:200], # Truncate
                    "success": result['success']
                })
            
            final_answer = f"""**Complex Word Problem - Decomposed**
**Original problem:** {query}
**Broken into {len(sub_problems)} sub-problems:**
"""
            for sp in sub_problems:
                final_answer += f"\n**Step {sp['step']}:** {sp['description']}\n"
                final_answer += f"Result: {sp['result']}\n"
            
            return {
                "decomposed": True,
                "sub_problems": sub_problems,
                "answer": final_answer,
                "method": "llm_guided_decomposition"
            }
        except Exception as e:
            self.logger.error(f"LLM decomposition failed: {e}")
            return None

    def _solve_constrained_optimization(self, query: str) -> Dict:
        """Placeholder for constrained optimization"""
        return {"error": "Constrained optimization not yet implemented"}
