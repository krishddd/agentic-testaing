"""
Knowledge Manager Module
Contains KnowledgeManager, AtomicConcept, and KnowledgeDomain.
"""
import logging
import json
import os
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

logger = logging.getLogger("reasoning_agent")

class KnowledgeDomain(Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    STATISTICS = "statistics"
    SOCIAL_SCIENCE = "social_science"
    GENERAL = "general"

@dataclass
class AtomicConcept:
    name: str
    topic: str
    definition: str
    formula: str
    keywords: List[str]
    examples: List[str]
    
    def to_document(self) -> Dict:
        """Convert to RAG-friendly document format"""
        content = f"**{self.name}**\n\n"
        content += f"**Definition:** {self.definition}\n"
        if self.formula:
            content += f"**Formula:** {self.formula}\n"
        content += f"**Examples:**\n" + "\n".join([f"- {ex}" for ex in self.examples])
        
        return {
            "page_content": content,
            "metadata": {
                "name": self.name,
                "topic": self.topic,
                "keywords": ", ".join(self.keywords)
            }
        }

class KnowledgeManager:
    """Manages externalized knowledge bases"""
    
    def __init__(self, data_path: str = "data/knowledge_base"):
        self.data_path = data_path
        self.bases: Dict[KnowledgeDomain, List[AtomicConcept]] = {}
        self._load_all()
    
    def _load_all(self):
        """Load concepts from JSON files"""
        # Load existing knowledge bases
        math_basic = self._load_from_json("math_concepts.json")
        math_advanced = self._load_from_json("advanced_math_concepts.json")
        self.bases[KnowledgeDomain.MATH] = math_basic + math_advanced
        
        physics_basic = self._load_from_json("physics_concepts.json")
        physics_advanced = self._load_from_json("advanced_physics_concepts.json")
        self.bases[KnowledgeDomain.PHYSICS] = physics_basic + physics_advanced
        
        # Load new knowledge bases
        self.bases[KnowledgeDomain.CHEMISTRY] = self._load_from_json("chemistry_concepts.json")
        self.bases[KnowledgeDomain.COMPUTER_SCIENCE] = self._load_from_json("computer_science_concepts.json")
        
        # Load structured data
        self.units = self._load_dict("units.json")
        self.physics_constants = self._load_dict("physics_constants.json")
        
        # Log loading stats
        logger.info(f"Loaded knowledge bases: Math={len(self.bases.get(KnowledgeDomain.MATH, []))}, "
                   f"Physics={len(self.bases.get(KnowledgeDomain.PHYSICS, []))}, "
                   f"Chemistry={len(self.bases.get(KnowledgeDomain.CHEMISTRY, []))}, "
                   f"CS={len(self.bases.get(KnowledgeDomain.COMPUTER_SCIENCE, []))}")

    def _load_dict(self, filename: str) -> Dict:
        """Load generic dictionary from JSON"""
        path = os.path.join(self.data_path, filename)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}

    def get_units_data(self) -> Dict:
        return self.units

    def get_physics_constants(self) -> Dict:
        return self.physics_constants

    
    def _load_from_json(self, filename: str) -> List[AtomicConcept]:
        """Load concepts from a specific JSON file"""
        path = os.path.join(self.data_path, filename)
        if not os.path.exists(path):
            # Try absolute path or adjust relative to current file if needed
            # For now just log warning and return empty
            logger.warning(f"Knowledge file not found: {path}")
            return []
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [AtomicConcept(**item) for item in data]
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
            
    def retrieve(self, query: str, domain: KnowledgeDomain = None, k: int = 3) -> List[Dict]:
        """Retrieve relevant concepts"""
        query_lower = query.lower()
        if domain:
            concepts = self.bases.get(domain, [])
        else:
            concepts = sum(self.bases.values(), [])
            
        scored = []
        
        for c in concepts:
            score = sum(3 for kw in c.keywords if kw in query_lower)
            score += 5 if c.name.lower() in query_lower else 0
            score += query_lower.count(c.topic.lower()) * 2
            if score > 0:
                scored.append((score, c))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c.to_document() for _, c in scored[:k]]
    
    def detect_domain(self, query: str) -> KnowledgeDomain:
        """Auto-detect domain from query with enhanced keyword matching"""
        q = query.lower()
        
        # Chemistry keywords
        if any(kw in q for kw in ['chemical', 'molecule', 'atom', 'bond', 'reaction', 'acid', 'base', 
                                    'pH', 'enthalpy', 'entropy', 'catalyst', 'oxidation', 'reduction',
                                    'electron', 'ion', 'compound', 'equilibrium constant', 'mole',
                                    'concentration', 'molarity', 'titration', 'redox', 'electrochemical']):
            return KnowledgeDomain.CHEMISTRY
        
        # Computer Science keywords
        if any(kw in q for kw in ['algorithm', 'data structure', 'tree', 'graph', 'sort', 'search',
                                    'complexity', 'big-o', 'hash', 'dynamic programming', 'recursion',
                                    'binary search', 'linked list', 'stack', 'queue', 'heap',
                                    'np-complete', 'polynomial time', 'dijkstra', 'bfs', 'dfs']):
            return KnowledgeDomain.COMPUTER_SCIENCE
        
        # Advanced Math keywords (differential equations, linear algebra, etc.)
        if any(kw in q for kw in ['differential equation', 'ode', 'pde', 'laplace transform',
                                    'fourier', 'eigenvector', 'eigenvalue', 'matrix', 'determinant',
                                    'vector space', 'linear transformation', 'group', 'ring', 'field',
                                    'homomorphism', 'topology', 'metric space', 'compact', 'convergence',
                                    'gradient', 'divergence', 'curl', 'line integral', 'surface integral']):
            return KnowledgeDomain.MATH
        
        # Basic Math keywords  
        if any(kw in q for kw in ['derivative', 'integral', 'solve', 'equation', 'x^', 'factor',
                                    'polynomial', 'quadratic', 'linear', 'calculus', 'algebra',
                                    'trigonometry', 'logarithm', 'exponential', 'limit']):
            return KnowledgeDomain.MATH
        
        # Advanced Physics keywords (quantum, statistical mechanics, etc.)
        if any(kw in q for kw in ['quantum', 'schrödinger', 'wavefunction', 'operator', 'hamiltonian',
                                    'eigenstate', 'spin', 'entanglement', 'boltzmann', 'partition function',
                                    'entropy', 'phase transition', 'fermi', 'bose', 'phonon', 'plasma',
                                    'lagrangian', 'hamiltonian mechanics', 'navier-stokes', 'fluid',
                                    'turbulence', 'reynolds', 'superconductor', 'ferromagnet']):
            return KnowledgeDomain.PHYSICS
        
        # Basic Physics keywords
        if any(kw in q for kw in ['velocity', 'force', 'mass', 'acceleration', 'energy', 'wave',
                                    'momentum', 'friction', 'gravity', 'electric', 'magnetic',
                                    'light', 'optics', 'thermodynamics', 'heat', 'temperature',
                                    'pressure', 'voltage', 'current', 'resistance']):
            return KnowledgeDomain.PHYSICS
        
        # Economics keywords
        if any(kw in q for kw in ['supply', 'demand', 'price', 'gdp', 'inflation', 'market',
                                    'economics', 'trade', 'fiscal', 'monetary']):
            return KnowledgeDomain.ECONOMICS
        
        # Statistics keywords  
        if any(kw in q for kw in ['mean', 'variance', 'probability', 'distribution', 'hypothesis',
                                    'statistics', 'correlation', 'regression', 'sample', 'population']):
            return KnowledgeDomain.STATISTICS
        
        # Psychology keywords
        if any(kw in q for kw in ['memory', 'cognitive', 'behavior', 'psychology', 'learning',
                                    'perception', 'emotion']):
            return KnowledgeDomain.PSYCHOLOGY
        
        # Social Science keywords
        if any(kw in q for kw in ['democracy', 'society', 'political', 'culture', 'social',
                                    'government', 'law']):
            return KnowledgeDomain.SOCIAL_SCIENCE
        
        return KnowledgeDomain.GENERAL
