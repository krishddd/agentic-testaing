"""
Universal LLM Gateway for Epistemic Agent

Provides a unified interface to multiple LLM providers using LiteLLM.
Supports Claude, OpenAI, Ollama, and 100+ other providers.

This enables the Active Inference framework to work with any AI agent
regardless of the underlying LLM provider.
"""

import asyncio
import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    TOGETHER = "together"
    GROQ = "groq"
    LOCAL = "local"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int]
    raw_response: Optional[Dict] = None
    finish_reason: str = "stop"
    

@dataclass 
class LLMConfig:
    """Configuration for LLM Gateway"""
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.2"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    fallback_providers: List[LLMProvider] = None
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = []


class EpistemicLLMGateway:
    """
    Universal LLM Gateway using LiteLLM for multi-provider support.
    
    Features:
    - Unified interface for 100+ LLM providers
    - Automatic fallback chain for reliability
    - OpenAI-compatible response format
    - Async/sync support
    - Built-in retry and error handling
    """
    
    # Provider to model prefix mapping for LiteLLM
    PROVIDER_PREFIXES = {
        LLMProvider.OLLAMA: "ollama/",
        LLMProvider.ANTHROPIC: "anthropic/",
        LLMProvider.OPENAI: "",  # OpenAI is default, no prefix
        LLMProvider.GOOGLE: "gemini/",
        LLMProvider.AZURE: "azure/",
        LLMProvider.TOGETHER: "together_ai/",
        LLMProvider.GROQ: "groq/",
        LLMProvider.LOCAL: "ollama/",
    }
    
    # Default models per provider
    DEFAULT_MODELS = {
        LLMProvider.OLLAMA: "llama3.2",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        LLMProvider.OPENAI: "gpt-4o",
        LLMProvider.GOOGLE: "gemini-1.5-pro",
        LLMProvider.GROQ: "llama-3.3-70b-versatile",
        LLMProvider.TOGETHER: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    }
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM Gateway.
        
        Args:
            config: LLMConfig with provider settings. If None, uses defaults from env.
        """
        self.config = config or self._load_config_from_env()
        self._litellm = None
        self._init_litellm()
        
    def _load_config_from_env(self) -> LLMConfig:
        """Load configuration from environment variables"""
        provider_str = os.getenv("LLM_PROVIDER", "ollama").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            logger.warning(f"Unknown provider {provider_str}, defaulting to Ollama")
            provider = LLMProvider.OLLAMA
            
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", self.DEFAULT_MODELS.get(provider, "llama3.2")),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("LLM_API_BASE") or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )
    
    def _init_litellm(self):
        """Initialize LiteLLM with configuration"""
        try:
            import litellm
            self._litellm = litellm
            
            # Configure LiteLLM
            litellm.drop_params = True  # Drop unsupported params gracefully
            litellm.set_verbose = False
            
            # Set API keys if available
            if self.config.api_key:
                if self.config.provider == LLMProvider.ANTHROPIC:
                    os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
                elif self.config.provider == LLMProvider.OPENAI:
                    os.environ["OPENAI_API_KEY"] = self.config.api_key
                    
            logger.info(f"LiteLLM gateway initialized for {self.config.provider.value}")
            
        except ImportError:
            logger.warning("LiteLLM not installed. Falling back to direct provider calls.")
            self._litellm = None
    
    def _get_model_string(self, provider: Optional[LLMProvider] = None, model: Optional[str] = None) -> str:
        """Get the LiteLLM model string with provider prefix"""
        provider = provider or self.config.provider
        model = model or self.config.model
        prefix = self.PROVIDER_PREFIXES.get(provider, "")
        return f"{prefix}{model}"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request to the configured LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Request JSON output format
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with standardized response
        """
        # Build messages list
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # Try primary provider, then fallbacks
        providers_to_try = [self.config.provider] + self.config.fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            try:
                response = await self._call_provider(
                    provider=provider,
                    messages=full_messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    json_mode=json_mode,
                    **kwargs
                )
                return response
                
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                last_error = e
                continue
        
        # All providers failed
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    async def _call_provider(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        **kwargs
    ) -> LLMResponse:
        """Call a specific provider"""
        
        if self._litellm:
            return await self._call_litellm(
                provider, messages, temperature, max_tokens, json_mode, **kwargs
            )
        else:
            return await self._call_direct(
                provider, messages, temperature, max_tokens, json_mode, **kwargs
            )
    
    async def _call_litellm(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        **kwargs
    ) -> LLMResponse:
        """Call LLM using LiteLLM"""
        model_string = self._get_model_string(provider)
        
        completion_kwargs = {
            "model": model_string,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.config.timeout,
        }
        
        # Add JSON mode if supported
        if json_mode:
            if provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
                completion_kwargs["response_format"] = {"type": "json_object"}
            elif provider == LLMProvider.OLLAMA:
                completion_kwargs["format"] = "json"
        
        completion_kwargs.update(kwargs)
        
        # Use asyncio.to_thread for sync LiteLLM calls
        response = await asyncio.to_thread(
            self._litellm.completion,
            **completion_kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=provider,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            finish_reason=response.choices[0].finish_reason or "stop"
        )
    
    async def _call_direct(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        **kwargs
    ) -> LLMResponse:
        """Direct provider calls when LiteLLM not available"""
        
        if provider == LLMProvider.OLLAMA:
            return await self._call_ollama(messages, temperature, max_tokens, json_mode)
        elif provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(messages, temperature, max_tokens, json_mode)
        elif provider == LLMProvider.OPENAI:
            return await self._call_openai(messages, temperature, max_tokens, json_mode)
        else:
            raise ValueError(f"Direct calls not supported for {provider.value}. Install LiteLLM.")
    
    async def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> LLMResponse:
        """Direct Ollama API call"""
        import ollama
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if json_mode:
            kwargs["format"] = "json"
        
        response = await asyncio.to_thread(ollama.chat, **kwargs)
        
        return LLMResponse(
            content=response["message"]["content"],
            model=self.config.model,
            provider=LLMProvider.OLLAMA,
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            },
            raw_response=response
        )
    
    async def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> LLMResponse:
        """Direct Anthropic API call"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.config.api_key)
        
        # Extract system message if present
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        
        response = await asyncio.to_thread(
            client.messages.create,
            model=self.config.model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_messages,
            temperature=temperature,
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider=LLMProvider.ANTHROPIC,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            finish_reason=response.stop_reason or "stop"
        )
    
    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> LLMResponse:
        """Direct OpenAI API call"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        client = OpenAI(api_key=self.config.api_key)
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=LLMProvider.OPENAI,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            finish_reason=response.choices[0].finish_reason or "stop"
        )
    
    # Convenience methods for common patterns
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Simple completion from a single prompt"""
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content
    
    async def complete_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict:
        """Completion with JSON output parsing"""
        import json
        
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            json_mode=True,
            **kwargs
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"error": "JSON parse failed", "raw": response.content}
    
    async def multi_sample(
        self,
        messages: List[Dict[str, str]],
        n_samples: int = 3,
        temperature: float = 0.8,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate multiple samples for uncertainty estimation.
        Used by the UncertaintyEstimator module.
        """
        tasks = [
            self.chat(messages, temperature=temperature, **kwargs)
            for _ in range(n_samples)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


# Singleton instance for easy access
_default_gateway: Optional[EpistemicLLMGateway] = None


def get_gateway(config: Optional[LLMConfig] = None) -> EpistemicLLMGateway:
    """Get or create the default LLM gateway instance"""
    global _default_gateway
    
    if _default_gateway is None or config is not None:
        _default_gateway = EpistemicLLMGateway(config)
    
    return _default_gateway
