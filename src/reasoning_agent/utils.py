"""
Utils Module
Contains UnitManager, ErrorRecoverySystem, ConfidenceEstimator, and AnswerVerifier.
"""
import logging
from typing import Dict, Tuple, Any, List
import re
import sympy as sp
from pint import UnitRegistry

logger = logging.getLogger("reasoning_agent")

class UnitManager:
    """Track and convert units"""
    
    def __init__(self, knowledge_manager=None):
        try:
            self.ureg = UnitRegistry()
            self.available = True
        except:
            self.available = False
            print("WARNING: pint not installed. Run: pip install pint")
        self.logger = logger
        self.knowledge = knowledge_manager
    
    
    def parse_value_with_unit(self, value_str: str) -> Tuple[float, str]:
        """Extract numeric value and unit from string like '5 kg' or '10 m/s'"""
        if not self.available:
            # Fallback: simple regex extraction
            match = re.match(r'([-+]?\d+\.?\d*)\s*([a-zA-Z/^²³]+)', value_str.strip())
            if match:
                return float(match.group(1)), match.group(2)
            return float(value_str), None
        
        try:
            quantity = self.ureg(value_str)
            return float(quantity.magnitude), str(quantity.units)
        except:
            # Try plain number
            try:
                return float(value_str), None
            except:
                return None, None
    
    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between units"""
        if not self.available:
            self.logger.warning("Unit conversion unavailable - returning original value")
            return value
        
        try:
            quantity = value * self.ureg(from_unit)
            converted = quantity.to(to_unit)
            return float(converted.magnitude)
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            return value
    
    def check_dimensional_consistency(self, units_dict: Dict[str, str],
                                     equation_type: str) -> Dict:
        """Check if units are dimensionally consistent"""
        if not self.available:
            return {"consistent": None, "reason": "Unit checking unavailable"}
        
        try:
            # Example: F = ma should have N = kg * m/s²
            if equation_type == "F=ma":
                F_dim = self.ureg(units_dict.get('F', 'N')).dimensionality
                m_dim = self.ureg(units_dict.get('m', 'kg')).dimensionality
                a_dim = self.ureg(units_dict.get('a', 'm/s^2')).dimensionality
                ma_dim = (self.ureg.kg * self.ureg.m / self.ureg.s**2).dimensionality
                
                consistent = (F_dim == ma_dim)
                return {
                    "consistent": consistent,
                    "F_dimensionality": str(F_dim),
                    "ma_dimensionality": str(ma_dim),
                    "message": "✓ Units consistent" if consistent else "✗ Unit mismatch!"
                }
        except Exception as e:
            return {"consistent": False, "error": str(e)}
        
        return {"consistent": None, "reason": "No rule for this equation type"}
    
    def standardize_physics_units(self, known_values: Dict[str, Any]) -> Dict:
        """Convert all physics values to SI units"""
        if not self.available:
            return known_values
        
        # Load from knowledge base if available
        si_units = {}
        if self.knowledge:
            units_data = self.knowledge.get_units_data()
            si_units = units_data.get('si_mappings', {})
        
        if not si_units:
            # Minimal fallback
            si_units = {
                'm': 'kg', 'F': 'N', 'a': 'm/s^2', 'v_i': 'm/s', 'v_f': 'm/s',
                't': 's', 'd': 'm', 'E': 'J', 'P': 'W', 'p': 'kg*m/s'
            }
        
        standardized = {}
        conversions = []
        
        for var, value in known_values.items():
            if isinstance(value, str):
                num_val, unit = self.parse_value_with_unit(value)
                if unit and var in si_units:
                    target_unit = si_units[var]
                    converted = self.convert(num_val, unit, target_unit)
                    standardized[var] = converted
                    conversions.append(f"{var}: {value} → {converted} {target_unit}")
                else:
                    standardized[var] = num_val if num_val else value
            else:
                standardized[var] = value
        
        return {
            "values": standardized,
            "conversions": conversions
        }

class ErrorRecoverySystem:
    """Handle and recover from errors gracefully"""
    
    def __init__(self, reasoning_engine):
        self.engine = reasoning_engine
        self.logger = logger
        self.retry_strategies = {
            "parsing": self._retry_parsing,
            "symbolic_math": self._retry_symbolic_math,
            "llm_generation": self._retry_llm_generation
        }
    
    def attempt_recovery(self, error_type: str, context: Dict) -> Dict:
        """Attempt to recover from error"""
        strategy = self.retry_strategies.get(error_type)
        if strategy:
            self.logger.info(f"Attempting recovery for {error_type}")
            return strategy(context)
        return {"recovered": False, "reason": f"No recovery strategy for {error_type}"}
    
    def _retry_parsing(self, context: Dict) -> Dict:
        """Retry parsing with different approaches"""
        problem = context.get('problem', '')
        
        strategies = [
            # Strategy 1: Simplify problem text
            lambda p: re.sub(r'\s+', ' ', p).strip(),
            # Strategy 2: Remove special characters
            lambda p: re.sub(r'[^\w\s=+\-*/^().,]', '', p),
            # Strategy 3: Extract just the mathematical expression
            lambda p: self._extract_core_expression(p)
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                simplified = strategy(problem)
                self.logger.info(f"Retry {i}: '{simplified}'")
                
                parsed = self.engine.parser._fallback_parse(simplified)
                if parsed.get('type') != 'qualitative': # Successfully parsed as math/physics
                    return {"recovered": True, "parsed": parsed, "strategy": f"simplification_{i}"}
            except Exception as e:
                continue
        
        return {"recovered": False, "reason": "All parsing strategies failed"}
    
    def _retry_symbolic_math(self, context: Dict) -> Dict:
        """Retry symbolic math with fallbacks"""
        operation = context.get('operation')
        expr_str = context.get('expression', '')
        
        # Strategy 1: Try numerical method
        if operation == 'integrate' and context.get('bounds'):
            numerical = self.engine.math_engine.numerical_solver.numerical_integration(
                expr_str,
                context['bounds']['lower'],
                context['bounds']['upper']
            )
            if 'error' not in numerical:
                return {"recovered": True, "result": numerical, "method": "numerical"}
        
        # Strategy 2: Simplify expression
        simplified_expr = expr_str.replace(' ', '').replace('**', '^')
        try:
            result = self.engine.math_engine.parse_expression(simplified_expr)
            if result:
                return {"recovered": True, "expression": result, "method": "simplified"}
        except:
            pass
        
        return {"recovered": False, "reason": "Symbolic math recovery failed"}
    
    def _retry_llm_generation(self, context: Dict) -> Dict:
        """Retry LLM generation with modified prompt"""
        original_prompt = context.get('prompt', '')
        
        # Try simpler, more direct prompt
        simplified_prompt = f"Briefly answer: {original_prompt}"
        
        try:
            response = self.engine.llm.generate(simplified_prompt)
            if response and "[LLM" not in response:
                return {"recovered": True, "response": response, "method": "simplified_prompt"}
        except:
            pass
        
        return {"recovered": False, "reason": "LLM retry failed"}
    
    def _extract_core_expression(self, problem: str) -> str:
        """Extract the core mathematical expression from text"""
        # Look for equations or expressions
        patterns = [
            r'=\s*(.+?)(?:\s|$)', # After equals sign
            r'(?:^|\s)([x-z]\^?\d.*?)(?:\s|$)', # Variables with operations
            r'(\d+[x-z]\^?\d+.*?)(?:\s|$)' # Numbers with variables
        ]
        
        for pattern in patterns:
            match = re.search(pattern, problem)
            if match:
                return match.group(1).strip()
        
        return problem

class ConfidenceEstimator:
    """Estimate confidence in answers"""
    
    def __init__(self):
        self.logger = logger
    
    def estimate_confidence(self, problem_type: str, parsed: Dict,
                          result: Dict, verification: Dict = None) -> Dict:
        """Calculate confidence score (0-1) with explanation"""
        confidence = 1.0
        factors = []
        
        # Factor 1: Parsing quality
        if not parsed.get('parse_success', True):
            confidence *= 0.7
            factors.append("Fallback parsing used (-30%)")
        
        # Factor 2: Computation success
        if 'error' in result:
            confidence *= 0.3
            factors.append("Computation error (-70%)")
        
        # Factor 3: Problem complexity
        if problem_type == 'math':
            subtype = parsed.get('subtype', '')
            if subtype in ['equation', 'derivative', 'integral']:
                confidence *= 0.95 # High confidence in basic operations
            elif subtype in ['system', 'limit']:
                confidence *= 0.85 # Medium confidence
            else:
                confidence *= 0.75 # Lower for undefined
        
        elif problem_type == 'physics':
            subtype = parsed.get('subtype', '')
            known = parsed.get('known_values', {})
            if len(known) >= 3:
                confidence *= 0.9
            elif len(known) == 2:
                confidence *= 0.8
                factors.append("Limited known values (-20%)")
            else:
                confidence *= 0.6
                factors.append("Insufficient known values (-40%)")
        
        elif problem_type == 'qualitative':
            # Lower confidence for qualitative without LLM
            confidence *= 0.75
            factors.append("Qualitative reasoning (baseline 75%)")
        
        # Factor 4: Verification
        if verification:
            if verification.get('verified', False):
                confidence = min(1.0, confidence * 1.1) # Boost
                factors.append("Solution verified (+10%)")
            else:
                confidence *= 0.5
                factors.append("Verification failed (-50%)")
        
        # Classify confidence level
        if confidence >= 0.9:
            level = "VERY HIGH"
        elif confidence >= 0.75:
            level = "HIGH"
        elif confidence >= 0.5:
            level = "MEDIUM"
        elif confidence >= 0.3:
            level = "LOW"
        else:
            level = "VERY LOW"
        
        return {
            "score": round(confidence, 2),
            "level": level,
            "factors": factors,
            "interpretation": self._interpret_confidence(confidence)
        }
    
    def _interpret_confidence(self, score: float) -> str:
        if score >= 0.9:
            return "This answer is highly reliable. You can trust it."
        elif score >= 0.75:
            return "This answer is likely correct, but double-check if critical."
        elif score >= 0.5:
            return "This answer may be correct, but verification recommended."
        elif score >= 0.3:
            return "This answer is uncertain. Manual verification strongly recommended."
        else:
            return "This answer is highly uncertain. Do not rely on it without verification."

class AnswerVerifier:
    """Verify solutions by substituting back into original problem"""
    
    def __init__(self, math_engine):
        self.math_engine = math_engine
        self.logger = logger
    
    def verify_equation_solution(self, equation_str: str, variable: str,
                                 solutions: List) -> Dict[str, Any]:
        """Verify solutions by substitution"""
        
        # Check if math engine has SymPy loaded
        try:
            sp.Symbol("x")
        except:
             return {"verified": False, "reason": "SymPy unavailable"}
        
        try:
            var = sp.Symbol(variable)
            if '=' in equation_str:
                left, right = equation_str.split('=', 1)
                left_expr = self.math_engine.parse_expression(left.strip())
                right_expr = self.math_engine.parse_expression(right.strip())
            else:
                left_expr = self.math_engine.parse_expression(equation_str)
                right_expr = 0
            
            verification_results = []
            for sol in solutions:
                try:
                    sol_expr = sp.sympify(sol)
                    left_sub = left_expr.subs(var, sol_expr)
                    left_num = left_sub.evalf()
                    
                    if isinstance(right_expr, (int, float)):
                        right_num = right_expr
                    else:
                        right_sub = right_expr.subs(var, sol_expr)
                        right_num = right_sub.evalf()
                    
                    # Check if equal within tolerance
                    diff = abs(complex(left_num) - complex(right_num))
                    is_valid = diff < 1e-10
                    
                    verification_results.append({
                        "solution": str(sol),
                        "valid": is_valid,
                        "left_value": str(left_num),
                        "right_value": str(right_num),
                        "difference": float(diff)
                    })
                except Exception as e:
                    verification_results.append({
                        "solution": str(sol),
                        "valid": False,
                        "error": str(e)
                    })
            
            all_valid = all(r.get("valid", False) for r in verification_results)
            return {
                "verified": all_valid,
                "results": verification_results,
                "summary": f"{'✓' if all_valid else '✗'} {len([r for r in verification_results if r.get('valid')])}/{len(solutions)} solutions verified"
            }
        except Exception as e:
            return {"verified": False, "error": str(e)}
