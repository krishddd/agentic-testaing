import os
import torch
import time
import subprocess
import shutil
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from sentence_transformers import CrossEncoder
from src.logging import logger

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class Model:
    def __init__(self, config_dict) -> None:
        logger.info("Initializing Model class")
        self.gemini_embed_model = config_dict['gemini_embedding_model']
        self.gemini_model = config_dict['gemini_model']
        self.ollama_embed_model = config_dict['ollama_embedding_model']
        self.ollama_model = config_dict['ollama_model']
        self.start_time = time.time()

    def log_model_loading_time(self, model_name):
        end_time = time.time()
        loading_time = end_time - self.start_time
        logger.info(f"{model_name} loading time: {loading_time:.2f} seconds")
        self.start_time = time.time()

    def _clear_model_cache(self, model_cache_path):
        """Helper method to aggressively clear model cache"""
        if not model_cache_path.exists():
            logger.info(f"Cache path doesn't exist: {model_cache_path}")
            return True
        
        success = False
        
        # Method 1: subprocess (most reliable for NFS/stale handles)
        try:
            logger.info(f"Attempting subprocess removal of {model_cache_path}")
            result = subprocess.run(
                ['rm', '-rf', str(model_cache_path)],
                capture_output=True,
                timeout=10,
                check=False
            )
            if result.returncode == 0:
                logger.info("✓ Subprocess removal successful")
                success = True
            else:
                logger.warning(f"Subprocess removal failed: {result.stderr.decode()}")
        except Exception as e:
            logger.warning(f"Subprocess method error: {e}")
        
        # Method 2: shutil.rmtree
        if model_cache_path.exists():
            try:
                logger.info("Attempting shutil removal")
                shutil.rmtree(model_cache_path, ignore_errors=True)
                if not model_cache_path.exists():
                    logger.info("✓ Shutil removal successful")
                    success = True
            except Exception as e:
                logger.warning(f"Shutil method error: {e}")
        
        # Method 3: Clear individual files (last resort)
        if model_cache_path.exists():
            try:
                logger.info("Attempting to remove individual files")
                for root, dirs, files in os.walk(model_cache_path, topdown=False):
                    for name in files:
                        try:
                            os.unlink(os.path.join(root, name))
                        except:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except:
                            pass
                # Try to remove the root directory
                try:
                    os.rmdir(model_cache_path)
                    logger.info("✓ Individual file removal successful")
                    success = True
                except:
                    pass
            except Exception as e:
                logger.warning(f"Individual file removal error: {e}")
        
        return success or not model_cache_path.exists()

    def load_cross_encoder_model(self, max_retries=3):
        """Load CrossEncoder with stale file handle protection"""
        model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        cache_dir = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        model_cache_path = Path(cache_dir) / 'hub' / f'models--{model_name.replace("/", "--")}'
        
        for attempt in range(max_retries):
            try:
                logger.info(f'Attempting to load CrossEncoder (attempt {attempt + 1}/{max_retries})')
                cross_encoder = CrossEncoder(model_name, device=device)
                
                # Verify it works
                test_score = cross_encoder.predict([["test", "document"]])
                logger.info(f'CrossEncoder model successfully loaded and verified')
                return cross_encoder
                
            except Exception as e:
                is_stale = 'Stale file handle' in str(e) or getattr(e, 'errno', None) == 116
                
                if is_stale and attempt < max_retries - 1:
                    logger.warning(f'Stale file handle detected, moving corrupted cache...')
                    
                    if model_cache_path.exists():
                        try:
                            backup = model_cache_path.parent / f"{model_cache_path.name}-corrupted-{int(time.time())}"
                            model_cache_path.rename(backup)
                            logger.info(f'Moved to: {backup.name}')
                            
                            # Async cleanup in background
                            subprocess.Popen(['rm', '-rf', str(backup)], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL)
                        except Exception as rename_error:
                            logger.warning(f'Cache move failed: {rename_error}')
                    
                    time.sleep(2 ** attempt)
                    continue
                
                if attempt == max_retries - 1:
                    logger.error(f'Failed to load CrossEncoder after {max_retries} attempts: {str(e)}')
                raise
        
        raise RuntimeError('Failed to load CrossEncoder')

    def load_gemini_embedding(self):
        try:
            embedding = GoogleGenerativeAIEmbeddings(
                model=self.gemini_embed_model,
                task_type="retrieval_document"
            )
            logger.info('Gemini Embedding successfully loaded')
            self.log_model_loading_time("Gemini Embedding")
            return embedding
        except Exception as e:
            logger.error(f'Error Gemini Embedding: {str(e)}')
            raise

    def load_gemini_model(self):
        try:
            model = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=0.1,
                max_tokens=512,
                timeout=None,
                max_retries=2,
                top_k=90,
                top_p=0.5
            )
            logger.info('Gemini Model successfully loaded')
            self.log_model_loading_time("Gemini Model")
            return model
        except Exception as e:
            logger.error(f'Error loading Gemini Model: {str(e)}')
            raise

    def load_ollama_embedding(self):
        try:
            embedding = OllamaEmbeddings(
                model=self.ollama_embed_model
            )
            logger.info('Ollama Embedding successfully loaded')
            return embedding
        except Exception as e:
            logger.error(f'Error Ollama Embedding: {str(e)}')
            raise

    def load_ollama_model(self):
        """Standard Ollama model configuration"""
        logger.info('Loading Ollama Model')
        try:
            model = ChatOllama(
                model=self.ollama_model,
                temperature=0.2,
                num_ctx=4096,
                timeout=None,
                max_retries=2,
                top_k=50,
                top_p=0.9
            )
            logger.info('Ollama Model successfully loaded')
            return model
        except Exception as e:
            logger.error(f'Error loading Ollama Model: {str(e)}')
            raise

    def load_ollama_model_enhanced(self):
        """Enhanced Ollama model configuration for comprehensive answers"""
        logger.info('Loading Enhanced Ollama Model')
        try:
            model = ChatOllama(
                model=self.ollama_model,
                temperature=0.3,
                num_ctx=8192,
                num_predict=2048,
                timeout=None,
                max_retries=2,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1
            )
            logger.info('Enhanced Ollama Model loaded with extended context window')
            return model
        except Exception as e:
            logger.error(f'Error loading Enhanced Ollama Model: {str(e)}')
            raise

    def __del__(self):
        total_time = time.time() - self.start_time
        logger.info(f"Total model initialization time: {total_time:.2f} seconds")