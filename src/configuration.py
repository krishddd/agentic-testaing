from configparser import ConfigParser, NoSectionError, NoOptionError, MissingSectionHeaderError
from dotenv import load_dotenv
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging
import sys
from logging.handlers import RotatingFileHandler

load_dotenv()
print("Attempted to load variables from .env file.")

class Configuration:
    def master_config(self):  
        config = ConfigParser()
        try:
            config.read('./config/master-config.properties')
            dic1 = dict(config.items('datapath'))
            dic2 = dict(config.items('model_type'))
            dic3 = dict(config.items('stages_flags'))
            master_dict = {**dic1, **dic2, **dic3}
            return master_dict
        except (NoSectionError, NoOptionError, MissingSectionHeaderError) as e:
            print(f"Error reading master configuration: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred in master_config: {e}")
            return {}

    def params_config(self):
        config = ConfigParser()
        try:
            config.read('./config/params-config.properties')
            dic1 = dict(config.items('data_params'))
            dic2 = dict(config.items('embedding'))
            dic3 = dict(config.items('model_params'))
            params_dict = {**dic1, **dic2, **dic3}
            return params_dict
        except (NoSectionError, NoOptionError, MissingSectionHeaderError) as e:
            print(f"Error reading params configuration: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred in params_config: {e}")
            return {}

class ConfigurationManager:
    def configurations(self):
        config_obj = Configuration()
        try:
            master_dict = config_obj.master_config()
            params_dict = config_obj.params_config()
            merged_dict = {**master_dict, **params_dict}
            return merged_dict
        except Exception as e:
            print(f"An error occurred while merging configurations: {e}")
            return {}
    
    def get_available_domains(self, base_path: str = "dataset") -> list:
        """
        Scan the base dataset directory and return list of domain folders.
        Each folder represents a domain-specific database.
        """
        try:
            base = Path(base_path)
            if not base.exists():
                print(f"Base path {base_path} does not exist")
                return []
            
            # Get all subdirectories
            domains = [d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith('.')]
            print(f"Found domains: {domains}")
            return sorted(domains)
        except Exception as e:
            print(f"Error scanning domains: {e}")
            return []
    
    def get_domain_config(self, domain_name: str) -> dict:
        """
        Get configuration for a specific domain.
        Creates domain-specific paths for data and vector store.
        """
        base_config = self.configurations()
        
        # Create domain-specific paths
        domain_config = base_config.copy()
        domain_config['domain_name'] = domain_name
        domain_config['data_path'] = f"dataset/{domain_name}"
        domain_config['db_vector_path'] = f"database/chromadb/{domain_name}"
        domain_config['collection_name'] = f"{domain_name}_documents"
        
        return domain_config

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    num_ctx: int = 8192
    num_predict: int = 2048
    max_reasoning_steps: int = 15
    log_level: str = "INFO"
    log_file: Optional[str] = "reasoning_agent.log"
    enable_sympy: bool = True
    enable_llm_parsing: bool = True

def setup_logging(config: AgentConfig) -> logging.Logger:
    logger = logging.getLogger("reasoning_agent")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    if config.log_file:
        try:
            fh = RotatingFileHandler(config.log_file, maxBytes=10485760, backupCount=3)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
        except: pass
    return logger