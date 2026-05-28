"""
Physics Solver Module
Contains PhysicsSolver2D.
"""
import logging
import math
from typing import Dict, Any

logger = logging.getLogger("reasoning_agent")

class PhysicsSolver2D:
    """Advanced 2D physics problems"""
    
    def __init__(self, knowledge_manager=None):
        self.logger = logger
        self.knowledge = knowledge_manager
        
        # Load constants
        self.g = 9.8 # default
        if self.knowledge:
            constants_data = self.knowledge.get_physics_constants()
            if constants_data and 'constants' in constants_data:
                g_data = constants_data['constants'].get('g')
                if g_data:
                    self.g = g_data.get('value', 9.8)
                    self.logger.info(f"Loaded physical constant g={self.g} from knowledge base")
    
    def solve_projectile_motion(self, known: Dict) -> Dict:
        """
        Solve projectile motion problems
        Known values can include: v_i, angle, h_max, range, t_flight, etc.
        """
        
        v_i = known.get('v_i') or known.get('v0') or known.get('initial_velocity')
        angle_deg = known.get('angle') or known.get('theta')
        h_max = known.get('h_max')
        range_val = known.get('range')
        t_flight = known.get('t_flight')
        
        results = {}
        steps = []
        
        if v_i and angle_deg:
            # We have initial velocity and angle - can calculate everything
            angle_rad = math.radians(angle_deg)
            v_x = v_i * math.cos(angle_rad)
            v_y = v_i * math.sin(angle_rad)
            
            steps.append(f"1. Initial velocity components:")
            steps.append(f" v_x = v_i × cos(θ) = {v_i} × cos({angle_deg}°) = {v_x:.4f} m/s")
            steps.append(f" v_y = v_i × sin(θ) = {v_i} × sin({angle_deg}°) = {v_y:.4f} m/s")
            
            # Time to reach max height
            t_max = v_y / self.g
            
            # Maximum height
            h_max_calc = (v_y ** 2) / (2 * self.g)
            results['h_max'] = h_max_calc
            steps.append(f"\n2. Maximum height:")
            steps.append(f" h_max = v_y² / (2g) = {v_y:.4f}² / (2 × {self.g}) = {h_max_calc:.4f} m")
            
            # Time of flight
            t_flight_calc = 2 * v_y / self.g
            results['t_flight'] = t_flight_calc
            steps.append(f"\n3. Time of flight:")
            steps.append(f" t = 2v_y / g = 2 × {v_y:.4f} / {self.g} = {t_flight_calc:.4f} s")
            
            # Range
            range_calc = v_x * t_flight_calc
            results['range'] = range_calc
            steps.append(f"\n4. Range:")
            steps.append(f" R = v_x × t = {v_x:.4f} × {t_flight_calc:.4f} = {range_calc:.4f} m")
            
            # Alternatively: R = v²sin(2θ)/g
            range_alt = (v_i ** 2 * math.sin(2 * angle_rad)) / self.g
            steps.append(f" (Verification: R = v² × sin(2θ) / g = {range_alt:.4f} m ✓)")
            
        elif v_i and h_max:
            # Can find angle
            # h_max = v_i² sin²(θ) / (2g)
            # sin²(θ) = 2g × h_max / v_i²
            sin_theta_sq = (2 * self.g * h_max) / (v_i ** 2)
            if sin_theta_sq > 1:
                return {"error": "Impossible: h_max too large for given v_i"}
            sin_theta = math.sqrt(sin_theta_sq)
            angle_rad = math.asin(sin_theta)
            angle_deg = math.degrees(angle_rad)
            results['angle'] = angle_deg
            steps.append(f"1. From h_max = v_i² sin²(θ) / (2g):")
            steps.append(f" sin²(θ) = 2gh_max / v_i² = {sin_theta_sq:.6f}")
            steps.append(f" θ = {angle_deg:.2f}°")
            
            # Now we can calculate other values
            v_x = v_i * math.cos(angle_rad)
            v_y = v_i * sin_theta
            t_flight_calc = 2 * v_y / self.g
            range_calc = v_x * t_flight_calc
            results['range'] = range_calc
            results['t_flight'] = t_flight_calc
            steps.append(f"\n2. Range: R = {range_calc:.4f} m")
            steps.append(f"3. Time of flight: t = {t_flight_calc:.4f} s")
        
        else:
            return {"error": "Insufficient information for projectile motion"}
        
        return {
            "type": "projectile_motion",
            "known": known,
            "results": results,
            "steps": steps,
            "formulas_used": [
                "v_x = v_i × cos(θ)",
                "v_y = v_i × sin(θ)",
                "h_max = v_y² / (2g)",
                "t_flight = 2v_y / g",
                "Range = v_x × t = v_i² × sin(2θ) / g"
            ]
        }
    
    def solve_incline_with_friction(self, known: Dict) -> Dict:
        """
        Solve incline problems with friction
        Known: m, angle, mu (friction coefficient), find: a
        """
        
        # Extract and validate inputs
        m = known.get('m')
        angle_deg = known.get('angle')
        mu = known.get('mu', 0.0)  # Default to 0 if not provided
        
        # Convert to numeric if needed
        try:
            m = float(m) if m is not None else None
            angle_deg = float(angle_deg) if angle_deg is not None else None
            mu = float(mu) if mu is not None else 0.0
        except (ValueError, TypeError):
            return {"error": "Invalid numeric values provided"}
        
        # Validate required values
        if m is None or angle_deg is None:
            return {"error": "Need mass and angle (friction coefficient optional, defaults to 0)"}
        
        angle_rad = math.radians(angle_deg)
        g = self.g
        
        steps = []
        steps.append("1. Forces acting on object:")
        steps.append(f" - Weight: W = mg = {m} × {g} = {m*g:.2f} N (downward)")
        steps.append(f" - Normal force: N = mg×cos(θ) (perpendicular to incline)")
        
        if mu > 0:
            steps.append(f" - Friction: f = μN (up the incline, μ={mu})")
        else:
            steps.append(f" - Friction: NONE (frictionless surface, μ=0)")
        
        # Force components
        F_parallel = m * g * math.sin(angle_rad)  # Down the incline
        N = m * g * math.cos(angle_rad)  # Normal force
        F_friction = mu * N  # Friction force
        
        steps.append(f"\n2. Force parallel to incline (downward):")
        steps.append(f" F_|| = mg×sin(θ) = {m}×{g}×sin({angle_deg}°) = {F_parallel:.4f} N")
        
        steps.append(f"\n3. Normal force:")
        steps.append(f" N = mg×cos(θ) = {m}×{g}×cos({angle_deg}°) = {N:.4f} N")
        
        steps.append(f"\n4. Friction force:")
        steps.append(f" f = μN = {mu}×{N:.4f} = {F_friction:.4f} N")
        
        # Net force
        F_net = F_parallel - F_friction
        
        steps.append(f"\n5. Net force down incline:")
        steps.append(f" F_net = F_|| - f = {F_parallel:.4f} - {F_friction:.4f} = {F_net:.4f} N")
        
        # Acceleration
        a = F_net / m
        
        steps.append(f"\n6. Acceleration:")
        steps.append(f" a = F_net / m = {F_net:.4f} / {m} = {a:.4f} m/s²")
        
        if a < 0:
            steps.append(f"\nNote: Negative acceleration means friction exceeds gravitational component")
            steps.append(f"Object will not slide (static friction holds it)")
        
        return {
            "type": "incline_with_friction",
            "known": known,
            "results": {
                "F_parallel": F_parallel,
                "F_friction": F_friction,
                "F_net": F_net,
                "acceleration": a
            },
            "steps": steps
        }

    def _parse_numeric_value(self, value: Any) -> float:
        """Extract numeric value from string or return float/int"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Regex to find the first floating point number
            import re
            match = re.search(r'-?\d+(\.\d+)?', value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    def solve_kinematics_1d(self, known: Dict, find: str) -> Dict:
        """
        Solve 1D kinematics problems with robust variable mapping.
        Handles verbose variable names (e.g., 'initial_velocity' -> 'v_i').
        """
        # 1. Variable Mapping
        map_to_standard = {
            "initial_velocity": "v_i", "v0": "v_i", "vi": "v_i", "velocity_initial": "v_i", "u": "v_i",
            "final_velocity": "v_f", "v": "v_f", "vf": "v_f", "velocity_final": "v_f",
            "acceleration": "a", "accel": "a",
            "time": "t", "duration": "t",
            "distance": "d", "displacement": "d", "x": "d", "delta_x": "d", "s": "d"
        }
        
        # Normalize 'known' dictionary
        standard_known = {}
        for k, v in known.items():
            key_lower = k.lower().replace(" ", "_")
            
            # Map key
            standard_key = None
            if key_lower in map_to_standard:
                standard_key = map_to_standard[key_lower]
            elif k in ['v_i', 'v_f', 'a', 't', 'd']:
                standard_key = k
            
            if standard_key:
                # Value parsing
                parsed_val = self._parse_numeric_value(v)
                if parsed_val is not None:
                    standard_known[standard_key] = parsed_val
                else:
                    self.logger.warning(f"Could not parse numeric value from '{v}' for key '{k}'")
                
        # Normalize 'find' target
        target = map_to_standard.get(find.lower().replace(" ", "_"), find)
        
        # Normalize 'find' target
        target = map_to_standard.get(find.lower().replace(" ", "_"), find)
        
        self.logger.info(f"Solving kinematics. Known: {standard_known}, Target: {target}")
        print(f"DEBUG: Solving kinematics. Known: {standard_known}, Target: {target}") # Debug print
        
        # 2. Extract values (None if missing)
        v_i = standard_known.get('v_i')
        v_f = standard_known.get('v_f')
        a = standard_known.get('a')
        t = standard_known.get('t')
        d = standard_known.get('d')
        

        # 3. Solver Logic (Kinematic Equations)
        # We need 3 knowns to find the 4th. 
        # Equations:
        # 1. v_f = v_i + a*t
        # 2. d = v_i*t + 0.5*a*t^2
        # 3. v_f^2 = v_i^2 + 2*a*d
        # 4. d = 0.5*(v_i + v_f)*t
        
        result = None
        equation_used = ""
        steps = []
        
        try:
            # Case 1: Find acceleration (a)
            if target == 'a':
                if v_i is not None and v_f is not None and t is not None:
                    # a = (v_f - v_i) / t
                    result = (v_f - v_i) / t
                    equation_used = "v_f = v_i + a*t  =>  a = (v_f - v_i) / t"
                elif v_i is not None and v_f is not None and d is not None:
                    # v_f^2 = v_i^2 + 2ad => a = (v_f^2 - v_i^2) / (2d)
                    result = (v_f**2 - v_i**2) / (2 * d)
                    equation_used = "v_f² = v_i² + 2ad  =>  a = (v_f² - v_i²) / (2d)"
                elif v_i is not None and t is not None and d is not None:
                    # d = v_i*t + 0.5*a*t^2 => d - v_i*t = 0.5*a*t^2 => a = 2(d - v_i*t) / t^2
                    result = 2 * (d - v_i * t) / (t**2)
                    equation_used = "d = v_i*t + ½at²  =>  a = 2(d - v_i*t) / t²"
            
            # Case 2: Find distance (d)
            elif target == 'd':
                if v_i is not None and t is not None and a is not None:
                    result = v_i * t + 0.5 * a * t**2
                    equation_used = "d = v_i*t + ½at²"
                elif v_i is not None and v_f is not None and t is not None:
                    result = 0.5 * (v_i + v_f) * t
                    equation_used = "d = ½(v_i + v_f)t"
                elif v_i is not None and v_f is not None and a is not None:
                    if a == 0:
                         result = "Undefined (division by zero acceleration)"
                    else:
                        result = (v_f**2 - v_i**2) / (2 * a)
                        equation_used = "v_f² = v_i² + 2ad => d = (v_f² - v_i²) / 2a"

            # Case 3: Find final velocity (v_f)
            elif target == 'v_f':
                if v_i is not None and a is not None and t is not None:
                    result = v_i + a * t
                    equation_used = "v_f = v_i + a*t"
                elif v_i is not None and a is not None and d is not None:
                    # v_f^2 = v_i^2 + 2ad
                    v_f_squared = v_i**2 + 2 * a * d
                    if v_f_squared < 0:
                        return {"error": "Impossible physical situation (negative velocity squared)"}
                    result = math.sqrt(v_f_squared)
                    equation_used = "v_f = √(v_i² + 2ad)"

            # Case 4: Find time (t)
            elif target == 't':
                if v_i is not None and v_f is not None and a is not None:
                    if a == 0:
                        if v_i == v_f: result = "Any time (constant velocity)"
                        else: result = "Impossible (change in velocity with 0 acceleration)"
                    else:
                        result = (v_f - v_i) / a
                        equation_used = "t = (v_f - v_i) / a"
            
            # Case 5: Find initial velocity (v_i)
            elif target == 'v_i':
                if v_f is not None and a is not None and t is not None:
                    result = v_f - a * t
                    equation_used = "v_i = v_f - a*t"
            
            if result is not None:
                steps.append(f"1. Identified knowns: {{k: v for k,v in standard_known.items() if v is not None}}")
                steps.append(f"2. Goal: Find {target}")
                steps.append(f"3. Selected equation: {equation_used}")
                steps.append(f"4. Calculated result: {result}")
                
                return {
                    "type": "kinematics_1d",
                    "known": known,
                    "target": find,
                    "results": {find: result},
                    "steps": steps,
                    "formula": equation_used
                }
            else:
                return {"error": f"Insufficient information to find '{find}' with available tools. Knowns: {standard_known}"}

        except Exception as e:
            self.logger.error(f"Kinematics error: {e}")
            return {"error": f"Calculation error: {str(e)}"}
