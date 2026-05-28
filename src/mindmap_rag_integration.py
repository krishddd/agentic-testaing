"""
Mindmap RAG Integration Module

Provides functions to generate mind maps from RAG-retrieved documents
with source attribution, cross-references, and quality metrics.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.advanced_mindmap import (
    AdvancedMindMapGenerator,
    MindMapResponse,
    MindMapNode,
    MindMapEdge,
    MindMapCluster,
    MindMapStatistics
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED MODELS FOR RAG-INTEGRATED MINDMAPS
# ============================================================================

class SourceAttribution(BaseModel):
    """Source attribution for mindmap nodes."""
    doc_id: str
    source_file: str
    page: Optional[str] = None
    confidence: float = Field(ge=0, le=1, default=0.8)
    excerpt: str = ""


class EnhancedMindMapNode(BaseModel):
    """Extended MindMapNode with source attribution."""
    id: str
    label: str
    level: int
    node_type: str = "concept"
    importance: int = Field(default=5, ge=1, le=10)
    color: Optional[str] = None
    icon: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    # NEW: Source attribution fields
    source_doc_id: Optional[str] = None
    source_attribution: Optional[SourceAttribution] = None
    confidence: float = Field(ge=0, le=1, default=0.8)


class MindMapQualityMetrics(BaseModel):
    """Quality metrics for the generated mindmap."""
    coverage_score: float = Field(ge=0, le=1, description="% of source content represented")
    coherence_score: float = Field(ge=0, le=1, description="Logical node relationships")
    balance_score: float = Field(ge=0, le=1, description="Hierarchy distribution balance")
    cross_reference_count: int = Field(ge=0, description="Number of cross-document links")
    overall_score: float = Field(ge=0, le=1)


class RAGMindMapResponse(BaseModel):
    """Response model for RAG-integrated mindmap generation."""
    mindmap: MindMapResponse
    source_documents: List[Dict[str, Any]] = Field(default_factory=list)
    quality_metrics: Optional[MindMapQualityMetrics] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# RAG-MINDMAP GENERATOR
# ============================================================================

class RAGMindMapGenerator:
    """
    Generates mind maps from RAG-retrieved documents with:
    - Source attribution for each node
    - Cross-document relationship detection
    - Quality metrics (coverage, coherence, balance)
    - Enhanced node metadata
    """
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        retriever: BaseRetriever,
        max_nodes: int = 50,
        max_depth: int = 5
    ):
        """
        Initialize the RAG-integrated mindmap generator.
        
        Args:
            llm: Language model for generation
            retriever: Retriever for document fetching
            max_nodes: Maximum nodes in mindmap
            max_depth: Maximum hierarchy depth
        """
        self.llm = llm
        self.retriever = retriever
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.base_generator = AdvancedMindMapGenerator(llm, max_nodes, max_depth)
        
        logger.info(f"Initialized RAGMindMapGenerator: max_nodes={max_nodes}, max_depth={max_depth}")
    
    def _retrieve_and_format_context(
        self, 
        query: str, 
        k: int = 10
    ) -> Tuple[str, List[Document], Dict[str, Document]]:
        """
        Retrieve documents and format as context with source tracking.
        
        Returns:
            Tuple of (formatted_context, documents, doc_map)
        """
        # Retrieve documents
        documents = self.retriever.invoke(query)
        
        if not documents:
            logger.warning(f"No documents retrieved for query: {query}")
            return "", [], {}
        
        # Build context with DOC markers for source tracking
        context_parts = []
        doc_map = {}
        
        for i, doc in enumerate(documents[:k], 1):
            doc_id = f"DOC{i}"
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            doc_map[doc_id] = doc
            
            context_parts.append(
                f"[{doc_id}] Source: {source}, Page: {page}\n"
                f"{doc.page_content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        logger.info(f"Retrieved {len(documents)} documents, using top {min(k, len(documents))}")
        
        return context, documents, doc_map
    
    def _extract_source_attributions(
        self, 
        mindmap: MindMapResponse, 
        doc_map: Dict[str, Document]
    ) -> List[SourceAttribution]:
        """
        Extract source attributions by analyzing node content against documents.
        """
        attributions = []
        
        for node in mindmap.nodes:
            if node.level == 0:  # Skip root
                continue
            
            # Find best matching document for this node
            best_match_id = None
            best_match_score = 0.0
            
            node_text = f"{node.label} {node.description or ''} {' '.join(node.keywords)}"
            node_text_lower = node_text.lower()
            
            for doc_id, doc in doc_map.items():
                doc_text_lower = doc.page_content.lower()
                
                # Simple keyword matching (could use embeddings for better matching)
                keywords_found = sum(1 for word in node_text_lower.split() 
                                    if len(word) > 3 and word in doc_text_lower)
                
                score = keywords_found / max(1, len(node_text_lower.split()))
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_id = doc_id
            
            if best_match_id and best_match_score > 0.2:
                doc = doc_map[best_match_id]
                attributions.append(SourceAttribution(
                    doc_id=best_match_id,
                    source_file=doc.metadata.get('source', 'Unknown'),
                    page=str(doc.metadata.get('page', '')),
                    confidence=min(1.0, best_match_score * 2),
                    excerpt=doc.page_content[:200]
                ))
        
        return attributions
    
    def _calculate_quality_metrics(
        self, 
        mindmap: MindMapResponse,
        documents: List[Document],
        attributions: List[SourceAttribution]
    ) -> MindMapQualityMetrics:
        """
        Calculate quality metrics for the mindmap.
        """
        # Coverage: ratio of documents represented in nodes
        docs_represented = len(set(a.doc_id for a in attributions))
        coverage = docs_represented / max(1, len(documents))
        
        # Coherence: ratio of nodes with proper parent connections
        node_ids = {n.id for n in mindmap.nodes}
        valid_edges = sum(1 for e in mindmap.edges 
                        if e.from_node in node_ids and e.to_node in node_ids)
        coherence = valid_edges / max(1, len(mindmap.edges))
        
        # Balance: evenness of node distribution across levels
        level_dist = mindmap.statistics.node_distribution if mindmap.statistics else {}
        if level_dist:
            values = list(level_dist.values())
            max_val = max(values) if values else 1
            min_val = min(values) if values else 1
            balance = min_val / max(1, max_val)
        else:
            balance = 0.5
        
        # Cross-reference count (edges between nodes from different sources)
        cross_refs = 0
        node_sources = {}
        for node, attr in zip(mindmap.nodes, attributions + [None] * len(mindmap.nodes)):
            if attr:
                node_sources[node.id] = attr.doc_id
        
        for edge in mindmap.edges:
            from_source = node_sources.get(edge.from_node)
            to_source = node_sources.get(edge.to_node)
            if from_source and to_source and from_source != to_source:
                cross_refs += 1
        
        overall = (coverage * 0.3 + coherence * 0.4 + balance * 0.3)
        
        return MindMapQualityMetrics(
            coverage_score=round(coverage, 2),
            coherence_score=round(coherence, 2),
            balance_score=round(balance, 2),
            cross_reference_count=cross_refs,
            overall_score=round(overall, 2)
        )
    
    def generate_from_query(
        self, 
        query: str,
        topic: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        depth_preference: str = "balanced",
        k: int = 10
    ) -> RAGMindMapResponse:
        """
        Generate a mindmap from RAG-retrieved documents.
        
        Args:
            query: Search query to retrieve relevant documents
            topic: Central topic for the mindmap (defaults to query)
            focus_areas: Optional areas to focus on
            depth_preference: "shallow", "balanced", or "deep"
            k: Number of documents to retrieve
            
        Returns:
            RAGMindMapResponse with mindmap, sources, and quality metrics
        """
        start_time = datetime.now()
        topic = topic or query
        
        logger.info(f"Generating RAG mindmap for query: {query[:100]}...")
        
        # Step 1: Retrieve and format context
        context, documents, doc_map = self._retrieve_and_format_context(query, k)
        
        if not context:
            logger.warning("No context retrieved, creating minimal mindmap")
            return RAGMindMapResponse(
                mindmap=self.base_generator._create_fallback_mindmap(topic, "No documents retrieved"),
                source_documents=[],
                quality_metrics=None,
                generation_metadata={
                    "query": query,
                    "error": "no_documents_retrieved"
                }
            )
        
        # Step 2: Generate mindmap from context
        mindmap = self.base_generator.generate_from_text(
            text_content=context,
            topic=topic,
            focus_areas=focus_areas,
            depth_preference=depth_preference
        )
        
        # Step 3: Extract source attributions
        attributions = self._extract_source_attributions(mindmap, doc_map)
        
        # Step 4: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(mindmap, documents, attributions)
        
        # Step 5: Prepare source document summaries
        source_documents = [
            {
                "doc_id": f"DOC{i+1}",
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "snippet": doc.page_content[:300] + "...",
                "rerank_score": doc.metadata.get('relevance_score', 'N/A')
            }
            for i, doc in enumerate(documents[:k])
        ]
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        logger.info(f"RAG mindmap generated: {mindmap.statistics.total_nodes} nodes, "
                   f"quality={quality_metrics.overall_score:.2f}, time={generation_time:.2f}s")
        
        return RAGMindMapResponse(
            mindmap=mindmap,
            source_documents=source_documents,
            quality_metrics=quality_metrics,
            generation_metadata={
                "query": query,
                "topic": topic,
                "focus_areas": focus_areas,
                "depth_preference": depth_preference,
                "documents_retrieved": len(documents),
                "documents_used": min(k, len(documents)),
                "generation_time_seconds": generation_time,
                "timestamp": end_time.isoformat()
            }
        )
    
    def export_to_mermaid_with_sources(self, response: RAGMindMapResponse) -> str:
        """
        Export mindmap to Mermaid format with source annotations.
        """
        lines = ["graph TD"]
        
        # Add nodes
        for node in response.mindmap.nodes:
            style = "(" if node.level == 0 else "["
            end_style = ")" if node.level == 0 else "]"
            label = node.label.replace('"', "'")
            lines.append(f'    {node.id}{style}"{label}"{end_style}')
        
        # Add edges
        for edge in response.mindmap.edges:
            arrow = "==>" if edge.weight > 1.5 else "-->"
            lines.append(f'    {edge.from_node} {arrow} {edge.to_node}')
        
        # Add styling
        for node in response.mindmap.nodes:
            if node.color:
                lines.append(f'    style {node.id} fill:{node.color},stroke:#333')
        
        # Add source legend as comment
        lines.append("")
        lines.append("%% Source Documents:")
        for doc in response.source_documents[:5]:
            lines.append(f"%%   {doc['doc_id']}: {doc['source']}")
        
        return '\n'.join(lines)
    
    def export_to_html_with_tooltips(self, response: RAGMindMapResponse) -> str:
        """
        Export mindmap as interactive HTML with source tooltips.
        """
        mermaid_code = self.export_to_mermaid_with_sources(response)
        
        # Build source tooltip data
        tooltips_json = {}
        for doc in response.source_documents:
            tooltips_json[doc['doc_id']] = {
                'source': doc['source'],
                'page': doc['page'],
                'snippet': doc['snippet'][:200]
            }
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG Mind Map: {response.generation_metadata.get('topic', 'Mind Map')}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        .metrics {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .metric {{
            background: #e8f5e9;
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .sources {{
            margin-top: 20px;
            padding: 15px;
            background: #fff3e0;
            border-radius: 8px;
        }}
        .source-item {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 {response.generation_metadata.get('topic', 'Mind Map')}</h1>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{response.mindmap.statistics.total_nodes if response.mindmap.statistics else 'N/A'}</div>
                <div class="metric-label">Nodes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{response.quality_metrics.coverage_score:.0%}</div>
                <div class="metric-label">Coverage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{response.quality_metrics.coherence_score:.0%}</div>
                <div class="metric-label">Coherence</div>
            </div>
            <div class="metric">
                <div class="metric-value">{response.quality_metrics.overall_score:.0%}</div>
                <div class="metric-label">Overall</div>
            </div>
        </div>
        
        <div class="mermaid">
{mermaid_code}
        </div>
        
        <div class="sources">
            <h3>📚 Source Documents</h3>
            {''.join(f'<div class="source-item"><strong>{doc["doc_id"]}</strong>: {doc["source"]} (Page {doc["page"]})</div>' for doc in response.source_documents[:10])}
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
        
        return html


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_mindmap_from_rag(
    query: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    topic: Optional[str] = None,
    max_nodes: int = 50,
    depth: str = "balanced"
) -> RAGMindMapResponse:
    """
    Convenience function to generate a mindmap from RAG retrieval.
    
    Args:
        query: Search query for document retrieval
        retriever: Document retriever
        llm: Language model
        topic: Central topic (defaults to query)
        max_nodes: Maximum nodes to generate
        depth: Depth preference
        
    Returns:
        RAGMindMapResponse with full results
    """
    generator = RAGMindMapGenerator(
        llm=llm,
        retriever=retriever,
        max_nodes=max_nodes
    )
    
    return generator.generate_from_query(
        query=query,
        topic=topic,
        depth_preference=depth
    )
