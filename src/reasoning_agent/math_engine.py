"""
Math Engine Module
Contains SymbolicMathEngine and NumericalSolver.
"""
import logging
import re
from typing import Dict, Any, List, Optional
import sympy as sp
from sympy import symbols, solve, diff, integrate, simplify, expand, factor, Eq
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# Optional imports handled gracefully
try:
    from scipy import optimize, integrate as scipy_integrate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed. Run: pip install scipy")

SYMPY_AVAILABLE = True # Assuming it is avail, but could add check

logger = logging.getLogger("reasoning_agent")

def clean_math_expression(expr_str: str) -> str:
    """Preprocess math string for Python evaluation"""
    expr_str = expr_str.strip()
    
    # 1. Replace ^ with **
    expr_str = expr_str.replace('^', '**')
    
    # 2. Add explicit multiplication for implicit cases like 2x -> 2*x
    # 10x -> 10*x
    expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)
    # )x -> )*x (e.g. (2+3)x)
    expr_str = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expr_str)
    # Removing letter-paren rule to avoid breaking functions like sin(x) -> sin*(x)
    # expr_str = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expr_str)
    # 2( -> 2*(
    expr_str = re.sub(r'(\d)(\()', r'\1*\2', expr_str)
    # )( -> )*(
    expr_str = re.sub(r'(\))(\()', r'\1*\2', expr_str)
    
    return expr_str

class NumericalSolver:
    """Numerical methods for problems without closed-form solutions"""
    
    def __init__(self):
        self.logger = logger
        self.scipy_available = SCIPY_AVAILABLE
        if self.scipy_available:
            self.optimize = optimize
            self.integrate_module = scipy_integrate
    
    def numerical_integration(self, func_str: str, lower: float, upper: float,
                             variable: str = 'x') -> Dict:
        """Numerical integration using scipy"""
        if not self.scipy_available:
            return {"error": "scipy not available"}
        
        try:
            import numpy as np
            # Create safe namespace
            safe_dict = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e,
                'x': 0 # placeholder
            }
            
            # Clean expression
            func_str = clean_math_expression(func_str)
            
            # Create lambda function safely
            func = eval(f"lambda {variable}: {func_str}", {"__builtins__": {}}, safe_dict)
            
            # Numerical integration
            result, error = self.integrate_module.quad(func, lower, upper)
            
            return {
                "method": "numerical_integration",
                "function": func_str,
                "bounds": {"lower": lower, "upper": upper},
                "result": result,
                "error_estimate": error,
                "steps": [
                    f"1. Function: f({variable}) = {func_str}",
                    f"2. Using scipy.integrate.quad (adaptive quadrature)",
                    f"3. Bounds: [{lower}, {upper}]",
                    f"4. Result: {result:.10f}",
                    f"5. Error estimate: ±{error:.2e}"
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def find_roots(self, func_str: str, variable: str = 'x',
                   initial_guess: float = 0) -> Dict:
        """Find roots numerically using Newton's method"""
        if not self.scipy_available:
            return {"error": "scipy not available"}
        
        try:
            import numpy as np
            safe_dict = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e
            }
            
            func_str = clean_math_expression(func_str)
            func = eval(f"lambda {variable}: {func_str}", {"__builtins__": {}}, safe_dict)
            
            # Try multiple initial guesses to find multiple roots
            guesses = [0, 1, -1, 2, -2, 10, -10]
            roots = []
            
            for guess in guesses:
                try:
                    root = self.optimize.fsolve(func, guess)[0]
                    # Check if this is actually a root
                    if abs(func(root)) < 1e-6:
                        # Check if we already found this root
                        if not any(abs(root - r) < 1e-4 for r in roots):
                            roots.append(root)
                except:
                    continue
            
            return {
                "method": "numerical_root_finding",
                "function": func_str,
                "roots": roots,
                "num_roots": len(roots),
                "steps": [
                    f"1. Function: f({variable}) = {func_str}",
                    f"2. Using scipy.optimize.fsolve (numerical solver)",
                    f"3. Tried multiple initial guesses",
                    f"4. Found {len(roots)} root(s): {[round(r, 6) for r in roots]}"
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_function(self, func_str: str, variable: str = 'x',
                          maximize: bool = True) -> Dict:
        """Find maximum or minimum of function"""
        if not self.scipy_available:
            return {"error": "scipy not available"}
        
        try:
            import numpy as np
            safe_dict = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e
            }
            
            func_str = clean_math_expression(func_str)
            func = eval(f"lambda {variable}: {func_str}", {"__builtins__": {}}, safe_dict)
            
            # If maximizing, minimize the negative
            if maximize:
                objective = lambda x: -func(x)
            else:
                objective = func
            
            # Try multiple starting points to find global optimum
            best_result = None
            best_value = float('inf') if not maximize else float('-inf')
            
            for x0 in [0, 1, -1, 2, -2, 5, -5, 10, -10]:
                try:
                    result = self.optimize.minimize(objective, x0, method='BFGS')
                    if result.success:
                        value = func(result.x[0])
                        if maximize:
                            if value > best_value:
                                best_value = value
                                best_result = result
                        else:
                            if value < best_value:
                                best_value = value
                                best_result = result
                except:
                    continue
            
            if best_result:
                optimal_x = best_result.x[0]
                optimal_y = func(optimal_x)
                
                return {
                    "method": "numerical_optimization",
                    "function": func_str,
                    "optimization_type": "maximize" if maximize else "minimize",
                    "optimal_x": optimal_x,
                    "optimal_y": optimal_y,
                    "steps": [
                        f"1. Function: f({variable}) = {func_str}",
                        f"2. Using scipy.optimize.minimize (BFGS method)",
                        f"3. Objective: {'maximize' if maximize else 'minimize'}",
                        f"4. Optimal point: {variable} = {optimal_x:.6f}",
                        f"5. Optimal value: f({optimal_x:.6f}) = {optimal_y:.6f}"
                    ]
                }
            else:
                return {"error": "Optimization failed to converge"}
                
        except Exception as e:
            return {"error": str(e)}

class SymbolicMathEngine:
    """Dynamic symbolic math using SymPy"""
    
    def __init__(self):
        self.logger = logger
        self.transformations = (standard_transformations + (implicit_multiplication_application,)) if SYMPY_AVAILABLE else None
        self.numerical_solver = NumericalSolver()
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
            """Parse string to SymPy expression with better preprocessing"""
            if not SYMPY_AVAILABLE:
                raise ValueError("SymPy not available")
            
            try:
                # Use shared cleaner first
                expr_str = clean_math_expression(expr_str)
                
                # Additional SymPy-specific preprocessing
                # 1. Convert e^(...) to exp(...) 
                expr_str = re.sub(r'\\be\\s*\\*\\*', 'exp', expr_str) # clean_math_expression handles ^ -> **
                
                # 2. Replace Unicode superscripts with ** notation
                superscript_map = {
                    '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5',
                    '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9'
                }
                for sup, replacement in superscript_map.items():
                    expr_str = expr_str.replace(sup, replacement)
                
                # 3. Parse using SymPy with transformations and local dict
                local_dict = {'E': sp.E, 'e': sp.E, 'exp': sp.exp, 'pi': sp.pi, 'I': sp.I}
                return parse_expr(expr_str, local_dict=local_dict, transformations=self.transformations)
                
            except Exception as e:
                self.logger.error(f"Parse error for '{expr_str}': {e}")
                
                # Try alternative parsing
                try:
                    local_dict = {'E': sp.E, 'e': sp.E, 'exp': sp.exp, 'pi': sp.pi}
                    return sp.sympify(expr_str, locals=local_dict)
                except:
                    raise ValueError(f"Cannot parse expression: {expr_str}")  
    
    def solve_equation(self, equation_str: str, variable: str = 'x') -> Dict[str, Any]:
        """Solve equation for variable with better error handling"""
        if not SYMPY_AVAILABLE:
            return {"error": "SymPy not available"}
        
        try:
            var = sp.Symbol(variable)
            
            # Preprocess equation string
            # Don't clean the whole equation string yet as it might contain '=' which cleaner ignores but better safe
            if '=' in equation_str:
                parts = equation_str.split('=')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Parse both sides (cleaner called inside parse_expression)
                    left_expr = self.parse_expression(left)
                    right_expr = self.parse_expression(right) if right and right != '0' else 0

                    
                    eq = Eq(left_expr, right_expr)
                else:
                    # Multiple = signs, try to parse as-is
                    eq = Eq(self.parse_expression(equation_str), 0)
            else:
                # No = sign, assume = 0
                eq = Eq(self.parse_expression(equation_str), 0)
            
            # Solve
            solutions = solve(eq, var)
            
            # Format solutions
            solutions_str = [str(s) for s in solutions]
            solutions_numeric = []
            
            for s in solutions:
                try:
                    if s.is_number:
                        numeric_val = complex(s.evalf())
                        # If imaginary part is negligible, show as real
                        if abs(numeric_val.imag) < 1e-10:
                            solutions_numeric.append(float(numeric_val.real))
                        else:
                            solutions_numeric.append(numeric_val)
                    else:
                        solutions_numeric.append(str(s))
                except:
                    solutions_numeric.append(str(s))
            
            return {
                "equation": str(eq),
                "variable": variable,
                "solutions": solutions_str,
                "solutions_numeric": solutions_numeric,
                "num_solutions": len(solutions),
                "steps": self._generate_solve_steps(eq, var, solutions)
            }
            
        except Exception as e:
            self.logger.error(f"Equation solving error: {e}")
            return {"error": str(e)}

    def integrate_expr(self, expr_str: str, variable: str = 'x', 
                        lower: float = None, upper: float = None) -> Dict[str, Any]:
        """Compute integral with numerical fallback"""
        if not SYMPY_AVAILABLE:
            if lower is not None and upper is not None:
                return self.numerical_solver.numerical_integration(expr_str, lower, upper, variable)
            return {"error": "SymPy not available"}
    
        try:
            var = sp.Symbol(variable)
            expr = self.parse_expression(expr_str)
            
            # Handle infinity bounds (convert string to SymPy infinity)
            lower_sym = lower
            upper_sym = upper
            
            if upper is not None and isinstance(upper, str):
                if upper.lower() in ['inf', 'infinity', '∞']:
                    upper_sym = sp.oo
            if lower is not None and isinstance(lower, str):
                if lower.lower() in ['-inf', '-infinity', '-∞']:
                    lower_sym = -sp.oo
            
            # Compute integral
            if lower_sym is not None and upper_sym is not None:
                result = integrate(expr, (var, lower_sym, upper_sym))
                integral_type = "definite"
            else:
                result = integrate(expr, var)
                integral_type = "indefinite"
            
            # Simplify and evaluate
            result_simplified = simplify(result)
            
            # Try numeric evaluation
            numeric_val = None
            if result_simplified.is_number or (integral_type == "definite"):
                try:
                    numeric_val = float(result_simplified.evalf())
                except:
                    pass
            
            # Format result string
            result_str = str(result_simplified)
            if integral_type == "indefinite":
                result_str += " + C"
            
            return {
                "original": str(expr),
                "variable": variable,
                "integral_type": integral_type,
                "result": result_str,
                "numeric": numeric_val,
                "bounds": {
                    "lower": str(lower) if lower is not None else None, 
                    "upper": str(upper) if upper is not None else None
                } if integral_type == "definite" else None
            }
            
        except Exception as e:
            # FALLBACK TO NUMERICAL for improper integrals
            if lower is not None and upper is not None:
                self.logger.warning(f"Symbolic integration failed, trying numerical: {e}")
                
                # Convert infinity strings to numeric bounds for approximation
                lower_num = lower if not isinstance(lower, str) else 0
                upper_num = upper
                
                if isinstance(upper, str) and upper.lower() in ['inf', 'infinity', '∞']:
                    upper_num = 50  # Often sufficient for e^(-x) type functions
                if isinstance(lower, str) and lower.lower() in ['-inf', '-infinity', '-∞']:
                    lower_num = -50
                
                numerical_result = self.numerical_solver.numerical_integration(
                    expr_str, lower_num, upper_num, variable
                )
                
                if "error" not in numerical_result:
                    numerical_result["note"] = "Symbolic integration failed, used numerical approximation for improper integral"
                    return numerical_result
            
            return {"error": str(e)}
        
    def optimize_expr(self, expr_str: str, variable: str = 'x',
                        maximize: bool = True) -> Dict[str, Any]:
            """Find maximum or minimum of expression"""
            if not SYMPY_AVAILABLE:
                # Use numerical optimization
                return self.numerical_solver.optimize_function(expr_str, variable, maximize)
            
            try:
                var = sp.Symbol(variable)
                expr = self.parse_expression(expr_str)
                
                # Find critical points by taking derivative and solving
                derivative = sp.diff(expr, var)
                critical_points = sp.solve(derivative, var)
                
                # Evaluate second derivative to classify critical points
                second_derivative = sp.diff(derivative, var)
                
                results = []
                for point in critical_points:
                    try:
                        if point.is_real:
                            y_value = expr.subs(var, point)
                            second_deriv_value = second_derivative.subs(var, point)
                            
                            # Classify
                            if second_deriv_value > 0:
                                point_type = "minimum"
                            elif second_deriv_value < 0:
                                point_type = "maximum"
                            else:
                                point_type = "inflection"
                            
                            results.append({
                                "x": float(point.evalf()),
                                "y": float(y_value.evalf()),
                                "type": point_type
                            })
                    except:
                        continue
                
                # Find the optimal based on maximize flag
                if results:
                    if maximize:
                        optimal = max(results, key=lambda r: r["y"])
                    else:
                        optimal = min(results, key=lambda r: r["y"])
                    
                    return {
                        "method": "symbolic_optimization",
                        "original": str(expr),
                        "variable": variable,
                        "optimization_type": "maximize" if maximize else "minimize",
                        "derivative": str(derivative),
                        "critical_points": results,
                        "optimal": optimal,
                        "steps": [
                            f"1. Function: f({variable}) = {expr}",
                            f"2. Derivative: f'({variable}) = {derivative}",
                            f"3. Critical points: {[r['x'] for r in results]}",
                            f"4. Optimal point: {variable} = {optimal['x']:.6f}, f({variable}) = {optimal['y']:.6f}"
                        ]
                    }
                else:
                    # No critical points found, try numerical
                    return self.numerical_solver.optimize_function(expr_str, variable, maximize)
                    
            except Exception as e:
                # Fallback to numerical
                self.logger.warning(f"Symbolic optimization failed, trying numerical: {e}")
                return self.numerical_solver.optimize_function(expr_str, variable, maximize)    

    def _generate_solve_steps(self, eq, var, solutions) -> List[str]:
        """Generate solution steps"""
        steps = [f"1. Original equation: {eq}"]
        lhs = eq.lhs - eq.rhs
        expanded = expand(lhs)
        if expanded != lhs:
            steps.append(f"2. Rearrange to standard form: {expanded} = 0")
        if len(solutions) == 1:
            steps.append(f"3. Solution: {var} = {solutions[0]}")
        else:
            for i, sol in enumerate(solutions):
                steps.append(f"3.{i+1}. Solution {i+1}: {var} = {sol}")
        return steps
    
    def differentiate(self, expr_str: str, variable: str = 'x', order: int = 1) -> Dict[str, Any]:
        """Compute derivative"""
        if not SYMPY_AVAILABLE:
            return {"error": "SymPy not available"}
        try:
            var = sp.Symbol(variable)
            expr = self.parse_expression(expr_str)
            result = diff(expr, var, order)
            return {
                "original": str(expr),
                "variable": variable,
                "order": order,
                "derivative": str(result),
                "simplified": str(simplify(result)),
                "steps": [
                    f"1. Original: f({variable}) = {expr}",
                    f"2. Apply differentiation rules",
                    f"3. Result: f{'′'*order}({variable}) = {simplify(result)}"
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def simplify_expr(self, expr_str: str) -> Dict[str, Any]:
        """Simplify expression"""
        if not SYMPY_AVAILABLE:
            return {"error": "SymPy not available"}
        try:
            expr = self.parse_expression(expr_str)
            return {
                "original": str(expr),
                "simplified": str(simplify(expr)),
                "expanded": str(expand(expr)),
                "factored": str(factor(expr))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def solve_system(self, equations: List[str], variables: List[str]) -> Dict[str, Any]:
        """Solve system of equations"""
        if not SYMPY_AVAILABLE:
            return {"error": "SymPy not available"}
        try:
            vars_sym = [sp.Symbol(v) for v in variables]
            eqs = []
            for eq_str in equations:
                if '=' in eq_str:
                    left, right = eq_str.split('=', 1)
                    eqs.append(Eq(self.parse_expression(left), self.parse_expression(right)))
                else:
                    eqs.append(Eq(self.parse_expression(eq_str), 0))
            solutions = solve(eqs, vars_sym)
            return {
                "equations": [str(eq) for eq in eqs],
                "variables": variables,
                "solutions": {str(k): str(v) for k, v in solutions.items()} if isinstance(solutions, dict) else str(solutions)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def compute_limit(self, expr_str: str, variable: str, point: str) -> Dict[str, Any]:
        """Compute limit"""
        if not SYMPY_AVAILABLE:
            return {"error": "SymPy not available"}
        try:
            var = sp.Symbol(variable)
            expr = self.parse_expression(expr_str)
            if point.lower() in ['inf', 'infinity', '∞']:
                pt = sp.oo
            elif point.lower() in ['-inf', '-infinity', '-∞']:
                pt = -sp.oo
            else:
                pt = self.parse_expression(point)
            result = sp.limit(expr, var, pt)
            return {
                "expression": str(expr),
                "variable": variable,
                "point": str(pt),
                "limit": str(result)
            }
        except Exception as e:
            return {"error": str(e)}

    # Physics-specific calculations
    def kinematic_solve(self, known: Dict[str, float], find: str) -> Dict[str, Any]:
        """Solve kinematics problems dynamically"""
        if not SYMPY_AVAILABLE:
            return self._kinematic_fallback(known, find)
        try:
            v_i, v_f, a, t, d = sp.symbols('v_i v_f a t d')
            equations = [
                Eq(v_f, v_i + a*t),
                Eq(d, v_i*t + sp.Rational(1,2)*a*t**2),
                Eq(v_f**2, v_i**2 + 2*a*d),
                Eq(d, sp.Rational(1,2)*(v_i + v_f)*t)
            ]
            var_map = {'v_i': v_i, 'v_f': v_f, 'a': a, 't': t, 'd': d}
            subs = {var_map[k]: v for k, v in known.items() if k in var_map}
            target = var_map.get(find)
            if not target:
                return {"error": f"Unknown variable: {find}"}
            
            for eq in equations:
                eq_sub = eq.subs(subs)
                try:
                    sol = solve(eq_sub, target)
                    if sol:
                        result = float(sol[0].evalf())
                        return {
                            "known": known,
                            "find": find,
                            "equation_used": str(eq),
                            "result": result,
                            "steps": [
                                f"1. Known values: {known}",
                                f"2. Using equation: {eq}",
                                f"3. Substituting values: {eq_sub}",
                                f"4. Solving for {find}: {result}"
                            ]
                        }
                except:
                    continue
            return {"error": "Could not solve with given values"}
        except Exception as e:
            return {"error": str(e)}
    
    def _kinematic_fallback(self, known: Dict, find: str) -> Dict:
        """Fallback kinematics without SymPy"""
        v_i = known.get('v_i', 0)
        v_f = known.get('v_f')
        a = known.get('a')
        t = known.get('t')
        d = known.get('d')
        
        if find == 'd' and v_i is not None and t is not None and a is not None:
            result = v_i * t + 0.5 * a * t**2
            return {"find": find, "result": result, "formula": "d = v_i*t + 0.5*a*t²"}
        elif find == 'v_f' and v_i is not None and a is not None and t is not None:
            result = v_i + a * t
            return {"find": find, "result": result, "formula": "v_f = v_i + a*t"}
        return {"error": "Cannot solve with given values"}
