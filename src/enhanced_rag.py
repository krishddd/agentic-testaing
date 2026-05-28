"""
Enhanced RAG Pipeline with Document Generator Features

This module extends the basic RAG pipeline with:
- Citation tracking with DOC markers
- Quality assessment (coherence, relevance, completeness)
- Multi-stage generation with iterative refinement
- Adaptive retrieval strategy
- Integration with document generator features
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from collections import defaultdict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.logging import logger
from src.model import Model
from src.retriever import RetrieverManager
from src.configuration import ConfigurationManager


# ============================================================================
# PYDANTIC MODELS FOR ENHANCED RAG
# ============================================================================

class CitationInfo(BaseModel):
    """Citation tracking information."""
    doc_id: str
    source: str
    page: Optional[str] = None
    content_hash: str
    cited_text: str
    position_in_output: int


class QualityMetrics(BaseModel):
    """Quality assessment metrics for generated responses."""
    coherence_score: float = Field(ge=0, le=1, default=0.5)
    relevance_score: float = Field(ge=0, le=1, default=0.5)
    completeness_score: float = Field(ge=0, le=1, default=0.5)
    citation_coverage: float = Field(ge=0, le=1, default=0.5)
    readability_score: float = Field(ge=0, le=1, default=0.5)
    overall_score: float = Field(ge=0, le=1, default=0.5)
    feedback: List[str] = Field(default_factory=list)


class EnhancedRAGResponse(BaseModel):
    """Enhanced response model for RAG with full metadata."""
    answer: str
    context_documents: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[CitationInfo] = Field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    generation_trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """Configuration for Enhanced RAG pipeline."""
    enable_citations: bool = True
    enable_quality_assessment: bool = False  # Opt-in for performance
    enable_multi_stage: bool = False
    max_iterations: int = 2
    document_type: Literal["chat_short", "chat_long", "detailed"] = "chat_short"
    include_source_snippets: bool = True


@dataclass
class RetrievalStrategy:
    """Adaptive retrieval strategy configuration."""
    initial_k: int = 10
    max_k: int = 20
    relevance_threshold: float = 0.6
    diversity_weight: float = 0.3


# ============================================================================
# ENHANCED RAG CLASS
# ============================================================================

class EnhancedRAG:
    """
    Enhanced RAG pipeline with document generator features.
    
    Features:
    - Citation tracking with DOC markers
    - Quality assessment for responses
    - Multi-stage generation with refinement
    - Adaptive retrieval strategy
    - Multiple document type support
    """
    
    # Prompts for different document types
    PROMPTS = {
        "chat_short": """You are a helpful assistant providing concise answers.

**CONTEXT DOCUMENTS**:
{context}

**QUESTION**: {input}

{citation_instruction}

Provide a clear, concise answer (2-4 paragraphs):""",
        
        "chat_long": """You are an expert assistant providing comprehensive, detailed answers.

**CONTEXT DOCUMENTS**:
{context}

**QUESTION**: {input}

{citation_instruction}

**INSTRUCTIONS**:
1. Provide a thorough, well-structured response
2. Cover multiple aspects of the topic
3. Include examples where relevant
4. Organize with clear sections if needed

**COMPREHENSIVE ANSWER**:""",
        
        "detailed": """You are an expert assistant specialized in providing comprehensive, well-cited answers.

**TASK**: Answer the question using the provided context with full detail.

**CRITICAL INSTRUCTIONS**:
1. Include ALL relevant information from context
2. Preserve structure (tables, lists, etc.)
3. Include specific terms, mechanisms, categories
4. Use sections and subsections for organization
5. Cite sources using [DOC1], [DOC2] markers
6. Do not summarize - provide FULL detail
7. Stay accurate - only use information from context

**CONTEXT DOCUMENTS**:
{context}

**QUESTION**: {input}

**DETAILED ANSWER**:"""
    }
    
    QUALITY_EVAL_PROMPT = """Evaluate this generated response for quality.

**QUESTION**: {question}
**RESPONSE**: {response}
**SOURCE CONTEXT**: {context}

Rate each criterion (0.0-1.0):
1. **Coherence**: Is it well-structured and logical?
2. **Relevance**: Does it address the question using relevant info?
3. **Completeness**: Are important aspects covered?
4. **Citation Coverage**: Are claims grounded in sources?
5. **Readability**: Is it clear and appropriate?

Respond in this format:
COHERENCE: [score]
RELEVANCE: [score]
COMPLETENESS: [score]
CITATION_COVERAGE: [score]
READABILITY: [score]

FEEDBACK:
- [Specific improvement 1]
- [Specific improvement 2]"""
    
    def __init__(
        self, 
        model: BaseChatModel, 
        retriever: BaseRetriever,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize the Enhanced RAG pipeline.
        
        Args:
            model: Language model for generation
            retriever: Retriever for fetching context
            config: Optional configuration settings
        """
        self.model = model
        self.retriever = retriever
        self.config = config or RAGConfig()
        self.chain_type = 'retriever'
        
        logger.info(f"Initialized EnhancedRAG with citations={self.config.enable_citations}, "
                   f"quality={self.config.enable_quality_assessment}")
    
    def _format_context_with_citations(
        self, 
        documents: List[Document]
    ) -> Tuple[str, Dict[str, Document]]:
        """Format documents with citation markers [DOC1], [DOC2], etc."""
        context_parts = []
        doc_map = {}
        
        for i, doc in enumerate(documents, 1):
            doc_id = f"DOC{i}"
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            # Store document reference
            doc_map[doc_id] = doc
            
            # Format with citation marker
            context_parts.append(
                f"[{doc_id}] Source: {source}, Page: {page}\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts), doc_map
    
    def _format_context_simple(self, documents: List[Document]) -> str:
        """Simple context formatting without citation markers."""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            parts.append(
                f"--- Document {i} (Source: {source}, Page: {page}) ---\n"
                f"{doc.page_content}\n"
            )
        return "\n".join(parts)
    
    def _extract_citations(
        self, 
        generated_content: str, 
        doc_map: Dict[str, Document]
    ) -> List[CitationInfo]:
        """Extract and validate citations from generated content."""
        citations = []
        
        # Find citation patterns like [DOC1], [DOC2], etc.
        pattern = r'\[DOC(\d+)\]'
        matches = re.finditer(pattern, generated_content)
        
        for match in matches:
            doc_num = match.group(1)
            doc_id = f"DOC{doc_num}"
            
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                
                # Extract context around citation
                start = max(0, match.start() - 100)
                end = min(len(generated_content), match.end() + 100)
                cited_text = generated_content[start:end]
                
                # Create content hash
                content_hash = hashlib.md5(
                    doc.page_content.encode()
                ).hexdigest()[:8]
                
                citation = CitationInfo(
                    doc_id=doc_id,
                    source=doc.metadata.get('source', 'Unknown'),
                    page=str(doc.metadata.get('page')) if doc.metadata.get('page') else None,
                    content_hash=content_hash,
                    cited_text=cited_text,
                    position_in_output=match.start()
                )
                citations.append(citation)
        
        return citations
    
    def _assess_quality(
        self, 
        question: str, 
        response: str, 
        context: str
    ) -> QualityMetrics:
        """Assess the quality of generated content."""
        try:
            eval_prompt = self.QUALITY_EVAL_PROMPT.format(
                question=question,
                response=response[:3000],  # Truncate for efficiency
                context=context[:2000]
            )
            
            llm_response = self.model.invoke(eval_prompt)
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Parse scores
            scores = {}
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    try:
                        scores[key] = float(value.strip())
                    except ValueError:
                        continue
            
            # Extract feedback
            feedback = []
            if 'FEEDBACK:' in content:
                feedback_section = content.split('FEEDBACK:')[1]
                feedback = [
                    line.strip('- ').strip()
                    for line in feedback_section.split('\n')
                    if line.strip().startswith('-')
                ]
            
            # Calculate overall score
            score_values = [v for k, v in scores.items() if isinstance(v, float)]
            overall = sum(score_values) / len(score_values) if score_values else 0.5
            
            return QualityMetrics(
                coherence_score=scores.get('coherence', 0.5),
                relevance_score=scores.get('relevance', 0.5),
                completeness_score=scores.get('completeness', 0.5),
                citation_coverage=scores.get('citation_coverage', 0.5),
                readability_score=scores.get('readability', 0.5),
                overall_score=overall,
                feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"Error assessing quality: {e}")
            return QualityMetrics(feedback=["Quality assessment unavailable"])
    
    def _refine_response(
        self, 
        question: str, 
        current_response: str, 
        context: str,
        quality_metrics: QualityMetrics
    ) -> str:
        """Refine response based on quality feedback."""
        refinement_prompt = f"""Improve this response based on quality feedback:

**ORIGINAL QUESTION**: {question}

**CURRENT RESPONSE**:
{current_response}

**QUALITY FEEDBACK**:
{chr(10).join(f"- {fb}" for fb in quality_metrics.feedback)}

**SCORES**:
- Overall: {quality_metrics.overall_score:.2f}
- Coherence: {quality_metrics.coherence_score:.2f}
- Relevance: {quality_metrics.relevance_score:.2f}
- Completeness: {quality_metrics.completeness_score:.2f}

**SOURCE CONTEXT**:
{context[:3000]}

**TASK**: Improve the response addressing the feedback. Maintain structure and enhance quality.

**IMPROVED RESPONSE**:"""
        
        response = self.model.invoke(refinement_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def invoke(
        self, 
        input_dict: dict,
        document_type: Optional[str] = None
    ) -> EnhancedRAGResponse:
        """
        Invoke the Enhanced RAG with full features.
        
        Args:
            input_dict: Dictionary with 'input' key containing the query
            document_type: Override document type from config
            
        Returns:
            EnhancedRAGResponse with answer, citations, quality metrics
        """
        query = input_dict.get('input', '')
        doc_type = document_type or self.config.document_type
        generation_trace = []
        
        logger.info(f"EnhancedRAG processing: {query[:100]}...")
        
        # Step 1: Retrieve documents
        retrieved_docs = self.retriever.invoke(query)
        generation_trace.append({
            "stage": "retrieval",
            "num_docs": len(retrieved_docs)
        })
        
        if not retrieved_docs:
            return EnhancedRAGResponse(
                answer="No relevant documents found for your query.",
                metadata={"error": "no_documents_retrieved"}
            )
        
        # Step 2: Format context with/without citations
        doc_map = {}
        if self.config.enable_citations:
            context, doc_map = self._format_context_with_citations(retrieved_docs)
            citation_instruction = "**CITATION REQUIREMENT**: Reference sources using [DOC1], [DOC2], etc."
        else:
            context = self._format_context_simple(retrieved_docs)
            citation_instruction = ""
        
        # Step 3: Get appropriate prompt
        prompt_template = self.PROMPTS.get(doc_type, self.PROMPTS["chat_short"])
        formatted_prompt = prompt_template.format(
            context=context,
            input=query,
            citation_instruction=citation_instruction
        )
        
        # Step 4: Generate response
        llm_response = self.model.invoke(formatted_prompt)
        answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        generation_trace.append({"stage": "generation", "iteration": 1})
        
        # Step 5: Extract citations (if enabled)
        citations = []
        if self.config.enable_citations:
            citations = self._extract_citations(answer, doc_map)
            generation_trace.append({
                "stage": "citation_extraction",
                "num_citations": len(citations)
            })
        
        # Step 6: Quality assessment (if enabled)
        quality_metrics = None
        if self.config.enable_quality_assessment:
            quality_metrics = self._assess_quality(query, answer, context)
            generation_trace.append({
                "stage": "quality_assessment",
                "overall_score": quality_metrics.overall_score
            })
            
            # Step 7: Multi-stage refinement (if enabled and quality is low)
            if self.config.enable_multi_stage and quality_metrics.overall_score < 0.7:
                for iteration in range(2, self.config.max_iterations + 1):
                    answer = self._refine_response(query, answer, context, quality_metrics)
                    quality_metrics = self._assess_quality(query, answer, context)
                    generation_trace.append({
                        "stage": "refinement",
                        "iteration": iteration,
                        "score": quality_metrics.overall_score
                    })
                    
                    if quality_metrics.overall_score >= 0.85:
                        break
        
        # Prepare context documents for response
        context_documents = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "rerank_score": doc.metadata.get('relevance_score', 'N/A'),
                "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            }
            for doc in retrieved_docs
        ]
        
        logger.info(f"EnhancedRAG completed: {len(answer)} chars, {len(citations)} citations")
        
        return EnhancedRAGResponse(
            answer=answer,
            context_documents=context_documents,
            citations=citations,
            quality_metrics=quality_metrics,
            generation_trace=generation_trace,
            metadata={
                "document_type": doc_type,
                "citations_enabled": self.config.enable_citations,
                "quality_assessment_enabled": self.config.enable_quality_assessment,
                "num_sources": len(retrieved_docs)
            }
        )
    
    def invoke_with_quality(self, query: str) -> EnhancedRAGResponse:
        """Convenience method: invoke with quality assessment enabled."""
        original_setting = self.config.enable_quality_assessment
        self.config.enable_quality_assessment = True
        
        result = self.invoke({"input": query})
        
        self.config.enable_quality_assessment = original_setting
        return result
    
    def invoke_with_refinement(
        self, 
        query: str, 
        max_iterations: int = 2
    ) -> EnhancedRAGResponse:
        """Convenience method: invoke with multi-stage refinement."""
        original_quality = self.config.enable_quality_assessment
        original_multi = self.config.enable_multi_stage
        original_iter = self.config.max_iterations
        
        self.config.enable_quality_assessment = True
        self.config.enable_multi_stage = True
        self.config.max_iterations = max_iterations
        
        result = self.invoke({"input": query}, document_type="detailed")
        
        self.config.enable_quality_assessment = original_quality
        self.config.enable_multi_stage = original_multi
        self.config.max_iterations = original_iter
        
        return result
    
    # Backward compatibility with original RAG interface
    @property
    def chain(self):
        """Return a chain-like interface for backward compatibility."""
        return self


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_enhanced_rag(config_dict: dict, config: Optional[RAGConfig] = None) -> EnhancedRAG:
    """
    Factory function to create EnhancedRAG with standard configuration.
    
    Args:
        config_dict: Configuration dictionary with model settings
        config: Optional RAGConfig for customization
        
    Returns:
        Configured EnhancedRAG instance
    """
    logger.info("Creating EnhancedRAG pipeline...")
    
    model_obj = Model(config_dict)
    embedding = model_obj.load_ollama_embedding()
    llm_model = model_obj.load_ollama_model()
    
    retriever_manager = RetrieverManager(embeddings=embedding, config_dict=config_dict)
    retriever = retriever_manager.get_retriever(
        k=15,
        enable_reranking=True,
        top_n_rerank=8
    )
    
    return EnhancedRAG(model=llm_model, retriever=retriever, config=config)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_dict = config_manager.configurations()
    
    # Create with default settings (citations on, quality off)
    rag = create_enhanced_rag(config_dict)
    
    # Test query
    question = "What are the key security mechanisms for autonomous agents?"
    
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # Standard invoke
    response = rag.invoke({"input": question})
    
    print(f"\nANSWER:\n{response.answer}")
    print(f"\nCITATIONS: {len(response.citations)}")
    for cit in response.citations:
        print(f"  - {cit.doc_id}: {cit.source}")
    print(f"\nSOURCES: {len(response.context_documents)}")
