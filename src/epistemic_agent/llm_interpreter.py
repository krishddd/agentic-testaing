import json
import logging
from typing import Dict, List, Optional
from .generative_model import BeliefState, FileStatus, UserIntent, RiskLevel
from .config import settings

logger = logging.getLogger(__name__)


class LLMBeliefInterpreter:
    """
    Uses LLM to dynamically interpret observations and update beliefs.
    Replaces hardcoded scenario patterns.
    
    Now uses the universal LLM Gateway for multi-provider support
    (Claude, OpenAI, Ollama, etc.)
    """
    
    def __init__(self, model_name: Optional[str] = None, gateway=None):
        """
        Initialize the belief interpreter.
        
        Args:
            model_name: Optional model name override
            gateway: Optional LLM Gateway instance (uses default if None)
        """
        self.model_name = model_name or settings.OLLAMA_MODEL
        self._gateway = gateway
        
        # Few-shot examples
        self.examples = [
            {
                "user_input": "Delete project.txt from documents",
                "observation": "Files found: ['project_v1.txt', 'project_v2.txt', 'project_final.txt']",
                "interpretation": {
                    "belief_update": {
                        "file_status": {"ambiguous": 0.9, "exists": 0.1},
                        "user_intent": {"delete": 0.95, "clarify": 0.05},
                        "risk_level": {"hazardous": 0.8, "moderate": 0.2}
                    },
                    "suggested_action": "ask_clarification",
                    "confidence": 0.85,
                    "reasoning": "Multiple matching files found for delete operation; user must specify which one"
                }
            },
            {
                "user_input": "List all Python files from src",
                "observation": "Files found: ['app.py', 'config.py', 'main.py', 'utils.py']",
                "interpretation": {
                    "belief_update": {
                        "file_status": {"exists": 0.95, "does_not_exist": 0.02, "ambiguous": 0.01, "unknown": 0.02},
                        "user_intent": {"read": 0.9, "unknown": 0.1},
                        "risk_level": {"safe": 0.95, "moderate": 0.05}
                    },
                    "suggested_action": "execute",
                    "confidence": 0.95,
                    "reasoning": "User asked to list files; multiple results IS the expected answer. Ready to present."
                }
            },
            {
                "user_input": "Plan a trip to Atlantis",
                "observation": "Search Results: No reliable sources found for city Atlantis",
                "interpretation": {
                    "belief_update": {
                        "file_status": {"does_not_exist": 0.95, "unknown": 0.05},
                        "user_intent": {"read": 0.8, "unknown": 0.2},
                        "risk_level": {"moderate": 0.6, "safe": 0.4}
                    },
                    "suggested_action": "abort",
                    "confidence": 0.9,
                    "reasoning": "Location appears fictional; cannot hallucinate travel info"
                }
            },
            {
                "user_input": "What is the capital of Japan?",
                "observation": "Search Results: [{'title': 'Japan', 'content': 'The capital of Japan is Tokyo...'}]",
                "interpretation": {
                    "belief_update": {
                        "file_status": {"exists": 0.9, "unknown": 0.1},
                        "user_intent": {"read": 0.95, "unknown": 0.05},
                        "risk_level": {"safe": 0.95, "moderate": 0.05}
                    },
                    "suggested_action": "execute",
                    "confidence": 0.95,
                    "reasoning": "Clear factual answer found from search results."
                }
            },
            {
                "user_input": "delete femp file from test_demo",
                "observation": "Files found: ['femp.txt', 'femp_2.txt', 'femp_3.txt']",
                "interpretation": {
                    "belief_update": {
                        "file_status": {"exists": 0.95, "ambiguous": 0.05},
                        "user_intent": {"delete": 0.95, "clarify": 0.05},
                        "risk_level": {"moderate": 0.7, "safe": 0.2, "hazardous": 0.1}
                    },
                    "suggested_action": "execute",
                    "confidence": 0.9,
                    "reasoning": "User explicitly said 'delete femp file'; femp.txt is the exact match. Safe to proceed with deletion."
                }
            }
        ]
    
    async def interpret_observation(
        self,
        user_input: str,
        observation: str,
        current_belief: BeliefState,
        action_history: List[str]
    ) -> Dict:
        """
        Dynamically interprets observation using LLM reasoning.
        
        Uses the universal LLM Gateway for multi-provider support.
        """
        prompt = self._build_interpretation_prompt(
            user_input, observation, current_belief, action_history
        )
        
        try:
            # Use LLM Gateway if available, fallback to direct Ollama
            if self._gateway is not None:
                gateway = self._gateway
            else:
                try:
                    from .llm_gateway import get_gateway
                    gateway = get_gateway()
                except ImportError:
                    # Fallback to direct Ollama for backward compatibility
                    return await self._interpret_with_ollama(prompt)
            
            # Call via gateway
            response = await gateway.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Bayesian belief state updater for an AI safety agent. Your role is to interpret observations and output valid probability distributions over hidden states.",
                json_mode=True
            )
            
            interpretation = json.loads(response.content)
            return self._validate_interpretation(interpretation)
            
        except Exception as e:
            logger.error(f"LLM interpretation error: {e}")
            return self._fallback_interpretation()
    
    async def _interpret_with_ollama(self, prompt: str) -> Dict:
        """Fallback direct Ollama call for backward compatibility"""
        import asyncio
        try:
            import ollama
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a Bayesian belief state updater for an AI safety agent.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                format='json'
            )
            content = response['message']['content']
            interpretation = json.loads(content)
            return self._validate_interpretation(interpretation)
        except Exception as e:
            logger.error(f"Ollama fallback error: {e}")
            return self._fallback_interpretation()
    
    def _build_interpretation_prompt(
        self,
        user_input: str,
        observation: str,
        current_belief: BeliefState,
        action_history: List[str]
    ) -> str:
        
        prompt = """Your task: Interpret the observation and update belief probabilities.

Current Context:
- User Request: {user_input}
- Observation: {observation}
- Previous Actions: {action_history}
- Current Belief: 
  * FileStatus: {file_status}
  * UserIntent: {intent}
  * RiskLevel: {risk}

State Space:
- FileStatus: exists, does_not_exist, ambiguous, unknown
- UserIntent: delete, read, clarify, unknown
- RiskLevel: safe, moderate, hazardous

Output JSON format:
{{
    "belief_update": {{
        "file_status": {{ "exists": 0.0, "does_not_exist": 0.0, "ambiguous": 0.0, "unknown": 0.0 }},
        "user_intent": {{ ... }},
        "risk_level": {{ ... }}
    }},
    "suggested_action": "continue_search|ask_clarification|execute|abort",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}
Ensure probabilities sum to 1.0 for each factor.
""".format(
            user_input=user_input,
            observation=observation,
            action_history=action_history[:5], # Limit history length in prompt
            file_status=current_belief.file_status_probs,
            intent=current_belief.user_intent_probs,
            risk=current_belief.risk_level_probs
        )
        
        # Add examples — pick the most relevant based on query type
        is_delete = any(w in user_input.lower() for w in ['delete', 'remove', 'erase', 'trash'])
        is_listing = any(w in user_input.lower() for w in ['list', 'show', 'what files', 'what is in'])
        is_factual = any(w in user_input.lower() for w in ['what', 'who', 'how', 'explain', '?'])
        
        if is_delete:
            # Show delete-success example first, then delete-ambiguity
            relevant_idx = [4, 0]  # delete-exact-match, then delete-ambiguity
        elif is_listing:
            # Show the list-success example first (most relevant)
            relevant_idx = [1, 0]  # list-success, then delete-ambiguity for contrast
        elif is_factual:
            relevant_idx = [3, 2]  # factual-success, then abort-fictional
        else:
            relevant_idx = [0, 2]  # delete-ambiguity, abort-fictional
        
        for idx in relevant_idx:
            if idx < len(self.examples):
                ex = self.examples[idx]
                prompt += f"\nExample:\nInput: {ex['user_input']}\nObservation: {ex['observation']}\nInterpretation: {json.dumps(ex['interpretation'])}\n"
        
        return prompt
    
    def _validate_interpretation(self, interpretation: Dict) -> Dict:
        """Validate and normalize LLM output"""
        # (Simplified validation logic)
        if 'belief_update' not in interpretation:
            return self._fallback_interpretation()
        return interpretation
    
    def _fallback_interpretation(self) -> Dict:
        return {
            "belief_update": {
                "file_status": {s.value: 1.0/len(FileStatus) for s in FileStatus},
                "user_intent": {s.value: 1.0/len(UserIntent) for s in UserIntent},
                "risk_level": {s.value: 1.0/len(RiskLevel) for s in RiskLevel}
            },
            "suggested_action": "ask_clarification",
            "confidence": 0.3,
            "reasoning": "Unable to interpret observation"
        }
