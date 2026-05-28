from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from typing import Optional, List
from enum import Enum, auto

# NOTE: RiskLevel enum removed from here to avoid collision with generative_model.RiskLevel.
# Use generative_model.RiskLevel instead.

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class Settings(BaseSettings):
    """
    Application configuration with robust validation.
    Reads from .env file and environment variables.
    """
    
    # Environment
    ENV: Environment = Field(default=Environment.DEVELOPMENT, env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # LLM Configuration (Ollama) — default aligned with Agentic_Testaing project
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="qwen3:8b", env="OLLAMA_MODEL")
    DEFAULT_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Agent Parameters
    CONFIDENCE_THRESHOLD: float = Field(default=0.9, ge=0.0, le=1.0, description="Minimum confidence to execute pragmatic actions")
    MAX_INFO_LOOP_ITERATIONS: int = Field(default=5, ge=1, le=20, description="Maximum epistemic actions before forced decision")
    
    # External APIs
    TAVILY_API_KEY: Optional[SecretStr] = Field(default=None, env="TAVILY_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

# Global settings instance
settings = Settings()
