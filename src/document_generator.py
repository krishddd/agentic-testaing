from typing import List, Dict, Any, Optional, Literal, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging
import re
from dataclasses import dataclass
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document generation types."""
    BRIEFING = "briefing"
    STUDY_GUIDE = "study_guide"
    BLOG_POST = "blog_post"
    TECHNICAL_REPORT = "technical_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    TUTORIAL = "tutorial"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    FAQ = "faq"
    NEWSLETTER = "newsletter"
    CHAT_SHORT = "chat_short"
    CHAT_LONG = "chat_long"


class QualityMetrics(BaseModel):
    """Quality assessment metrics for generated documents."""
    coherence_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    citation_coverage: float = Field(ge=0, le=1)
    readability_score: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    feedback: List[str] = Field(default_factory=list)


class CitationInfo(BaseModel):
    """Citation tracking information."""
    doc_id: str
    source: str
    page: Optional[str]
    content_hash: str
    cited_text: str
    position_in_output: int


class GenerationConfig(BaseModel):
    """Enhanced configuration for document generation."""
    document_type: DocumentType
    topic: str
    persona: Optional[str] = "expert assistant"
    tone: Optional[str] = "professional"
    length: Optional[Literal["short", "medium", "long"]] = "medium"
    target_audience: Optional[str] = "general"
    language_level: Optional[Literal["beginner", "intermediate", "advanced"]] = "intermediate"
    include_citations: bool = True
    include_visual_suggestions: bool = False
    adaptive_retrieval: bool = True
    multi_stage_generation: bool = False
    max_iterations: int = 1
    additional_instructions: Optional[str] = None
    
    @validator('max_iterations')
    def validate_iterations(cls, v):
        if v < 1 or v > 5:
            raise ValueError("max_iterations must be between 1 and 5")
        return v


class GeneratedDocument(BaseModel):
    """Enhanced response model for generated documents."""
    document_type: str
    content: str
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]
    citations: List[CitationInfo] = Field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    visual_suggestions: List[str] = Field(default_factory=list)
    alternative_sections: Dict[str, str] = Field(default_factory=dict)
    generation_trace: List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class RetrievalStrategy:
    """Adaptive retrieval strategy configuration."""
    initial_k: int = 5
    max_k: int = 15
    relevance_threshold: float = 0.6
    diversity_weight: float = 0.3
    recency_weight: float = 0.1


class EnhancedDocumentGenerator:
    """
    Advanced document generator with multi-stage generation, quality assessment,
    adaptive retrieval, and intelligent refinement capabilities.
    """
    
    def __init__(self, llm, retriever, embedding_model=None):
        """
        Initialize the enhanced document generator.
        
        Args:
            llm: Language model for generation
            retriever: Retriever for fetching relevant context
            embedding_model: Optional embedding model for similarity scoring
        """
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.prompts = self._initialize_prompts()
        self.quality_evaluator = self._initialize_quality_evaluator()
        
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize prompts for different document types with enhanced templates."""
        
        briefing_prompt = PromptTemplate.from_template("""
You are a strategic analyst creating a comprehensive briefing document.

**TASK**: Create a professional briefing document on: "{topic}"
**TARGET AUDIENCE**: {target_audience}
**LANGUAGE LEVEL**: {language_level}

**CONTEXT DOCUMENTS**:
{context}

**STRUCTURE YOUR BRIEFING AS FOLLOWS**:

# EXECUTIVE SUMMARY
• 3-5 high-impact bullet points capturing critical insights
• Each bullet should be actionable and specific

# SITUATION ANALYSIS
## Current State
Synthesize the present context with supporting evidence

## Key Trends & Patterns
Identify emerging patterns from the source materials

## Critical Issues
Highlight challenges, gaps, or areas requiring attention

# DETAILED FINDINGS
Break down major themes with clear subheadings:
- Use evidence from multiple sources
- Cross-reference contradictory information
- Quantify claims where possible

# STRATEGIC IMPLICATIONS
Analyze what these findings mean for stakeholders

# RECOMMENDATIONS
Priority-ranked actionable recommendations with:
- Expected impact
- Implementation complexity
- Resource requirements

# APPENDIX
{citation_instruction}

**QUALITY STANDARDS**:
- Every major claim must be grounded in source material
- Use clear, precise language
- Avoid speculation; flag uncertainties
- Structure for quick scanning and deep reading

{additional_instructions}

Generate the briefing document:
""")

        study_guide_prompt = PromptTemplate.from_template("""
You are an expert educator creating a comprehensive, learner-centered study guide.

**TOPIC**: {topic}
**TARGET AUDIENCE**: {target_audience}
**DIFFICULTY LEVEL**: {language_level}

**SOURCE MATERIALS**:
{context}

**CREATE A STUDY GUIDE WITH THESE SECTIONS**:

# LEARNING ROADMAP
## Overview & Objectives
- What will learners master?
- Why does this topic matter?
- Prerequisites (if any)

## Estimated Study Time
Realistic time investment for each section

# CORE CONCEPTS EXPLAINED
For each major concept:
### [Concept Name]
**Definition**: Clear, concise explanation
**Why It Matters**: Real-world relevance
**Key Example**: Concrete illustration
**Common Misconceptions**: What students often get wrong
**Memory Aid**: Mnemonic or mental model

# PROGRESSIVE QUIZ SECTIONS

## Foundational Questions (10 questions)
Test basic understanding with short-answer questions

## Application Questions (5 questions)
Scenario-based problems requiring concept application

## Advanced Synthesis Questions (3 questions)
Multi-concept integration challenges

# COMPLETE ANSWER KEY
Detailed answers with explanations for all quiz questions

# DEEP-DIVE ESSAY PROMPTS
5 thought-provoking essay questions that encourage:
- Critical analysis
- Cross-concept synthesis
- Real-world application

# COMPREHENSIVE GLOSSARY
Alphabetically organized key terms with:
- Clear definitions
- Context of use
- Related terms

# STUDY STRATEGIES
Personalized tips for mastering this material:
- Active recall techniques
- Spaced repetition schedule
- Connection to prior knowledge

{citation_instruction}

{additional_instructions}

Generate the study guide:
""")

        blog_post_prompt = PromptTemplate.from_template("""
You are a skilled content creator crafting an engaging, shareable blog post.

**TOPIC**: {topic}
**TARGET AUDIENCE**: {target_audience}
**TONE**: {tone}

**SOURCE MATERIALS**:
{context}

**CREATE A COMPELLING BLOG POST**:

# [Magnetic Headline That Promises Clear Value]
Use power words, numbers, or provocative questions

## Opening Hook (2-3 short paragraphs)
Start with:
- A surprising statistic or counterintuitive fact
- A relatable pain point or common mistake
- An intriguing question that demands answers

Make readers think: "I need to keep reading."

## The Big Picture (1 paragraph)
Why this topic matters RIGHT NOW

## [5-7 Core Insights - Each as H2]

For each major insight:
### **[Benefit-Driven Subheading]**

**The Insight**: Clearly state the key point (1-2 sentences)

**Why It's Surprising/Important**: Context and implications (2-3 sentences)

**Practical Application**: How readers can use this (2-3 sentences)

> **Pro Tip**: Actionable bonus advice in callout format

**Supporting Evidence**: Cite interesting findings from sources
{citation_instruction}

Use:
- Short paragraphs (2-4 sentences max)
- Bullet points for lists
- **Bold** for key phrases
- Conversational transitions

## Conclusion: The Path Forward
- Synthesize the key insights
- End with a memorable takeaway or call-to-action
- Leave readers with a powerful question to ponder

## Key Takeaways Box
3-5 bullet points summarizing main lessons

{visual_instruction}

{additional_instructions}

Generate the blog post:
""")

        technical_report_prompt = PromptTemplate.from_template("""
You are a technical writer creating a rigorous, detailed technical report.

**TOPIC**: {topic}
**TARGET AUDIENCE**: {target_audience}

**SOURCE MATERIALS**:
{context}

**STRUCTURE**:

# ABSTRACT
200-word summary of objectives, methodology, findings, and conclusions

# 1. INTRODUCTION
## 1.1 Background & Context
## 1.2 Problem Statement
## 1.3 Objectives & Scope
## 1.4 Methodology Overview

# 2. THEORETICAL FRAMEWORK
Relevant theories, models, and prior research

# 3. DETAILED ANALYSIS
## 3.1 [Major Topic Area 1]
### Technical Details
### Data/Evidence Analysis
### Interpretation

## 3.2 [Major Topic Area 2]
[Continue pattern]

# 4. RESULTS & DISCUSSION
## 4.1 Key Findings
## 4.2 Implications
## 4.3 Limitations
## 4.4 Comparison with Existing Work

# 5. CONCLUSIONS
## 5.1 Summary
## 5.2 Recommendations
## 5.3 Future Research Directions

# REFERENCES
{citation_instruction}

# APPENDICES (if needed)
Technical details, additional data, glossary

{additional_instructions}

Generate the technical report:
""")

        comparative_analysis_prompt = PromptTemplate.from_template("""
You are an analyst creating a balanced comparative analysis.

**TOPIC**: {topic}
**TARGET AUDIENCE**: {target_audience}

**SOURCE MATERIALS**:
{context}

**STRUCTURE**:

# EXECUTIVE SUMMARY
Key comparison insights in 4-5 bullets

# COMPARISON FRAMEWORK
## Evaluation Criteria
List the dimensions being compared

## Methodology
How comparisons were made

# DETAILED COMPARISON

For each major item/approach being compared:

## [Item/Approach Name]

### Overview
Brief description

### Strengths
- Advantage 1 (with evidence)
- Advantage 2 (with evidence)

### Weaknesses
- Limitation 1 (with evidence)
- Limitation 2 (with evidence)

### Best Use Cases
When this option excels

### Cost/Complexity Profile

# HEAD-TO-HEAD COMPARISON TABLE
Create a comparison matrix for easy scanning

# SYNTHESIS & RECOMMENDATIONS
## Overall Assessment
## Context-Specific Recommendations
When to choose each option

{citation_instruction}

{additional_instructions}

Generate the comparative analysis:
""")

        faq_prompt = PromptTemplate.from_template("""
You are creating a comprehensive, user-friendly FAQ document.

**TOPIC**: {topic}
**TARGET AUDIENCE**: {target_audience}

**SOURCE MATERIALS**:
{context}

**STRUCTURE**:

# Introduction
Brief overview of what this FAQ covers

# [Category 1: Basics]

## Q: [Common beginner question]?
**A**: Clear, concise answer with example if helpful

[Continue with 5-8 questions per category]

# [Category 2: Common Issues]
[Continue pattern]

# [Category 3: Advanced Topics]
[Continue pattern]

# Still Have Questions?
Where to find additional help

**GUIDELINES**:
- Anticipate actual user questions
- Use simple, jargon-free language
- Provide actionable answers
- Include examples where helpful
- Cross-reference related questions
{citation_instruction}

{additional_instructions}

Generate the FAQ document:
""")

        tutorial_prompt = PromptTemplate.from_template("""
You are creating a hands-on, step-by-step tutorial.

**TOPIC**: {topic}
**TARGET AUDIENCE**: {target_audience}
**SKILL LEVEL**: {language_level}

**SOURCE MATERIALS**:
{context}

**STRUCTURE**:

# Tutorial: [Descriptive Title]

## What You'll Learn
- Specific skill 1
- Specific skill 2
- Specific skill 3

## Prerequisites
What knowledge/tools are needed

## Estimated Time
Realistic completion time

# Part 1: Fundamentals

## Step 1: [Action Verb - Specific Task]
**Goal**: What this step accomplishes

**Instructions**:
1. First sub-step with clear action
2. Second sub-step
3. Third sub-step

**Expected Result**: What success looks like

**Troubleshooting**: Common issues and fixes

## Step 2: [Next Task]
[Continue pattern]

# Part 2: Building on Basics
[Continue with progressive difficulty]

# Part 3: Advanced Techniques
[Continue pattern]

# Practice Challenges
3-5 exercises to reinforce learning

# Next Steps
Where to go from here to continue learning

{citation_instruction}

{additional_instructions}

Generate the tutorial:
""")

        return {
            DocumentType.BRIEFING: briefing_prompt,
            DocumentType.STUDY_GUIDE: study_guide_prompt,
            DocumentType.BLOG_POST: blog_post_prompt,
            DocumentType.TECHNICAL_REPORT: technical_report_prompt,
            DocumentType.COMPARATIVE_ANALYSIS: comparative_analysis_prompt,
            DocumentType.FAQ: faq_prompt,
            DocumentType.TUTORIAL: tutorial_prompt,
            DocumentType.CHAT_SHORT: PromptTemplate.from_template("""
You are {persona} providing a concise, helpful response.

**QUESTION/TOPIC**: {topic}

**CONTEXT**:
{context}

**INSTRUCTIONS**:
Provide a brief, direct response (2-4 paragraphs) that:
- Directly addresses the question
- Uses the most relevant information from context
- Is conversational and easy to understand
- Avoids unnecessary detail

{additional_instructions}

Your response:
"""),
            DocumentType.CHAT_LONG: PromptTemplate.from_template("""
You are {persona} providing a comprehensive, thoughtful response.

**QUESTION/TOPIC**: {topic}

**CONTEXT**:
{context}

**INSTRUCTIONS**:
Provide a thorough response that:
- Covers multiple facets of the topic
- Explains concepts with examples
- Maintains a natural, conversational flow
- Organizes information logically
- Addresses potential follow-up questions

{additional_instructions}

Your comprehensive response:
""")
        }
    
    def _initialize_quality_evaluator(self) -> PromptTemplate:
        """Initialize quality evaluation prompt."""
        return PromptTemplate.from_template("""
You are a quality assessor evaluating a generated document.

**DOCUMENT TYPE**: {document_type}
**TOPIC**: {topic}

**GENERATED CONTENT**:
{content}

**SOURCE CONTEXT**:
{context}

**EVALUATION CRITERIA**:

1. **Coherence** (0-1): Is the document well-structured and logically organized?
2. **Relevance** (0-1): Does it directly address the topic using relevant information?
3. **Completeness** (0-1): Are all important aspects covered?
4. **Citation Coverage** (0-1): Are claims properly grounded in source material?
5. **Readability** (0-1): Is it clear and appropriate for the target audience?

**PROVIDE SCORES AND FEEDBACK**:

Respond in this exact format:
COHERENCE: [score]
RELEVANCE: [score]
COMPLETENESS: [score]
CITATION_COVERAGE: [score]
READABILITY: [score]

FEEDBACK:
- [Specific improvement suggestion 1]
- [Specific improvement suggestion 2]
- [Specific improvement suggestion 3]
""")
    
    def adaptive_retrieve_context(
        self, 
        query: str, 
        strategy: RetrievalStrategy,
        config: GenerationConfig
    ) -> List[Document]:
        """
        Adaptive retrieval that adjusts based on document type and quality needs.
        
        Implements:
        - Progressive retrieval (start small, expand if needed)
        - Diversity-aware selection
        - Relevance filtering
        """
        logger.info(f"Adaptive retrieval for query: {query}")
        
        # Initial retrieval
        initial_docs = self.retriever.invoke(query)
        
        if not initial_docs:
            return []
        
        # Score and filter documents
        scored_docs = self._score_documents(initial_docs, query, strategy)
        
        # Select diverse, relevant subset
        selected_docs = self._select_diverse_documents(
            scored_docs,
            strategy.initial_k,
            strategy.diversity_weight
        )
        
        # For complex document types, potentially expand
        complex_types = [
            DocumentType.BRIEFING,
            DocumentType.TECHNICAL_REPORT,
            DocumentType.STUDY_GUIDE,
            DocumentType.COMPARATIVE_ANALYSIS
        ]
        
        if config.document_type in complex_types and config.adaptive_retrieval:
            if len(selected_docs) < strategy.max_k:
                # Get more documents if initial set seems insufficient
                additional = self._get_complementary_documents(
                    selected_docs,
                    scored_docs,
                    strategy.max_k - len(selected_docs)
                )
                selected_docs.extend(additional)
        
        logger.info(f"Retrieved {len(selected_docs)} documents after adaptive selection")
        return selected_docs
    
    def _score_documents(
        self,
        documents: List[Document],
        query: str,
        strategy: RetrievalStrategy
    ) -> List[Tuple[Document, float]]:
        """Score documents based on relevance, diversity, and recency."""
        scored = []
        
        for doc in documents:
            # Base relevance score (from retriever if available)
            relevance = doc.metadata.get('relevance_score', 0.5)
            
            # Recency bonus (if date available)
            recency_score = self._calculate_recency_score(doc)
            
            # Combined score
            final_score = (
                relevance * (1 - strategy.recency_weight) +
                recency_score * strategy.recency_weight
            )
            
            scored.append((doc, final_score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _calculate_recency_score(self, doc: Document) -> float:
        """Calculate recency score for a document."""
        # Placeholder - implement based on your metadata structure
        # Could parse dates from metadata
        return 0.5
    
    def _select_diverse_documents(
        self,
        scored_docs: List[Tuple[Document, float]],
        k: int,
        diversity_weight: float
    ) -> List[Document]:
        """Select diverse documents using MMR-like approach."""
        if not scored_docs or k == 0:
            return []
        
        selected = []
        remaining = scored_docs.copy()
        
        # Always take the top document
        selected.append(remaining[0][0])
        remaining.pop(0)
        
        # Select remaining documents balancing relevance and diversity
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = -float('inf')
            
            for idx, (doc, relevance) in enumerate(remaining):
                # Calculate diversity from already selected
                diversity = self._calculate_diversity(doc, selected)
                
                # Combined score
                score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected.append(remaining[best_idx][0])
            remaining.pop(best_idx)
        
        return selected
    
    def _calculate_diversity(self, doc: Document, selected: List[Document]) -> float:
        """Calculate diversity score (how different doc is from selected)."""
        if not selected:
            return 1.0
        
        # Simple diversity based on source overlap
        doc_source = doc.metadata.get('source', '')
        selected_sources = {d.metadata.get('source', '') for d in selected}
        
        if doc_source in selected_sources:
            return 0.3  # Lower diversity if from same source
        return 1.0  # High diversity if from different source
    
    def _get_complementary_documents(
        self,
        selected: List[Document],
        all_scored: List[Tuple[Document, float]],
        additional_k: int
    ) -> List[Document]:
        """Get complementary documents that fill gaps."""
        # Get documents not yet selected
        selected_ids = {id(doc) for doc in selected}
        remaining = [
            doc for doc, score in all_scored
            if id(doc) not in selected_ids
        ]
        
        return remaining[:additional_k]
    
    def format_context_with_citations(
        self,
        documents: List[Document]
    ) -> Tuple[str, Dict[str, Document]]:
        """Format documents with citation markers."""
        context_parts = []
        doc_map = {}
        
        for i, doc in enumerate(documents, 1):
            doc_id = f"DOC{i}"
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            doc_map[doc_id] = doc
            
            context_parts.append(
                f"[{doc_id}] Source: {source}, Page: {page}\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(context_parts), doc_map
    
    def extract_citations(
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
                    page=str(doc.metadata.get('page')) if doc.metadata.get('page') is not None else None,  # Convert to string
                    content_hash=content_hash,
                    cited_text=cited_text,
                    position_in_output=match.start()
                )
                citations.append(citation)
        
        return citations
    
    def assess_quality(
        self,
        generated_content: str,
        topic: str,
        document_type: str,
        context: str
    ) -> QualityMetrics:
        """Assess the quality of generated content."""
        try:
            eval_prompt = self.quality_evaluator.format(
                document_type=document_type,
                topic=topic,
                content=generated_content[:3000],  # Truncate for efficiency
                context=context[:2000]
            )
            
            response = self.llm.invoke(eval_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
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
            score_values = [v for k, v in scores.items() if k != 'feedback']
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
            # Return default metrics
            return QualityMetrics(
                coherence_score=0.5,
                relevance_score=0.5,
                completeness_score=0.5,
                citation_coverage=0.5,
                readability_score=0.5,
                overall_score=0.5,
                feedback=["Quality assessment unavailable"]
            )
    
    def suggest_visuals(
        self,
        content: str,
        document_type: DocumentType
    ) -> List[str]:
        """Suggest visual elements that would enhance the document."""
        suggestions = []
        
        # Analyze content for visual opportunities
        if 'comparison' in content.lower() or document_type == DocumentType.COMPARATIVE_ANALYSIS:
            suggestions.append("📊 Comparison table or matrix chart")
        
        if any(word in content.lower() for word in ['process', 'step', 'workflow']):
            suggestions.append("🔄 Process flowchart or timeline")
        
        if any(word in content.lower() for word in ['trend', 'growth', 'increase', 'decrease']):
            suggestions.append("📈 Line or bar chart showing trends")
        
        if 'relationship' in content.lower() or 'correlation' in content.lower():
            suggestions.append("🔗 Network diagram or scatter plot")
        
        if document_type == DocumentType.STUDY_GUIDE:
            suggestions.append("🧠 Mind map for concept relationships")
            suggestions.append("📝 Flashcard designs for key terms")
        
        if document_type == DocumentType.TUTORIAL:
            suggestions.append("📸 Screenshots or diagrams for key steps")
            suggestions.append("✅ Checklist graphic")
        
        # Count numbers/statistics
        numbers = re.findall(r'\d+%|\d+\.\d+|\$\d+', content)
        if len(numbers) >= 3:
            suggestions.append("📊 Infographic highlighting key statistics")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def generate_document(
        self,
        config: GenerationConfig,
        retrieved_docs: Optional[List[Document]] = None,
        strategy: Optional[RetrievalStrategy] = None
    ) -> GeneratedDocument:
        """
        Generate document with enhanced features.
        
        Implements:
        - Adaptive retrieval
        - Citation tracking
        - Quality assessment
        - Visual suggestions
        - Multi-stage generation (if enabled)
        """
        logger.info(f"Generating {config.document_type} on: {config.topic}")
        
        generation_trace = []
        
        # Retrieval phase
        if retrieved_docs is None:
            strategy = strategy or RetrievalStrategy()
            retrieved_docs = self.adaptive_retrieve_context(
                config.topic,
                strategy,
                config
            )
            generation_trace.append({
                "stage": "retrieval",
                "num_docs": len(retrieved_docs),
                "strategy": "adaptive"
            })
        
        if not retrieved_docs:
            raise ValueError("No relevant context found for document generation")
        
        # Format context with citation support
        if config.include_citations:
            context, doc_map = self.format_context_with_citations(retrieved_docs)
            citation_instruction = "\n**CITATION REQUIREMENT**: Reference sources using [DOC1], [DOC2], etc. markers"
        else:
            context = self._format_context_simple(retrieved_docs)
            doc_map = {}
            citation_instruction = ""
        
        # Visual suggestions instruction
        visual_instruction = ""
        if config.include_visual_suggestions:
            visual_instruction = "\n**NOTE**: Suggest places where visuals would enhance understanding"
        
        # Get prompt template
        prompt_template = self.prompts.get(config.document_type)
        if not prompt_template:
            raise ValueError(f"Unsupported document type: {config.document_type}")
        
        # Prepare prompt inputs
        prompt_inputs = {
            "topic": config.topic,
            "context": context,
            "target_audience": config.target_audience,
            "language_level": config.language_level,
            "tone": config.tone,
            "persona": config.persona,
            "citation_instruction": citation_instruction,
            "visual_instruction": visual_instruction,
            "additional_instructions": config.additional_instructions or ""
        }
        
        # Multi-stage generation
        if config.multi_stage_generation and config.max_iterations > 1:
            generated_content = self._multi_stage_generate(
                prompt_template,
                prompt_inputs,
                config,
                context,
                generation_trace
            )
        else:
            # Single-stage generation
            formatted_prompt = prompt_template.format(**prompt_inputs)
            response = self.llm.invoke(formatted_prompt)
            generated_content = response.content if hasattr(response, 'content') else str(response)
            generation_trace.append({"stage": "generation", "iteration": 1})
        
        # Extract citations
        citations = []
        if config.include_citations:
            citations = self.extract_citations(generated_content, doc_map)
            generation_trace.append({
                "stage": "citation_extraction",
                "num_citations": len(citations)
            })
        
        # Quality assessment
        quality_metrics = self.assess_quality(
            generated_content,
            config.topic,
            config.document_type.value,
            context
        )
        generation_trace.append({
            "stage": "quality_assessment",
            "overall_score": quality_metrics.overall_score
        })
        
        # Visual suggestions
        visual_suggestions = []
        if config.include_visual_suggestions:
            visual_suggestions = self.suggest_visuals(
                generated_content,
                config.document_type
            )
        
        # Prepare source metadata
        sources = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "snippet": doc.page_content[:200] + "...",
                "relevance": doc.metadata.get('relevance_score', 'N/A')
            }
            for doc in retrieved_docs
        ]
        
        result = GeneratedDocument(
            document_type=config.document_type.value,
            content=generated_content,
            metadata={
                "topic": config.topic,
                "persona": config.persona,
                "tone": config.tone,
                "length": config.length,
                "target_audience": config.target_audience,
                "language_level": config.language_level,
                "num_sources": len(retrieved_docs),
                "generation_method": "multi_stage" if config.multi_stage_generation else "single_stage"
            },
            sources=sources,
            citations=citations,
            quality_metrics=quality_metrics,
            visual_suggestions=visual_suggestions,
            generation_trace=generation_trace
        )
        
        logger.info(f"Successfully generated {config.document_type} (Quality: {quality_metrics.overall_score:.2f})")
        return result
    
    def _format_context_simple(self, documents: List[Document]) -> str:
        """Simple context formatting without citation markers."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(
                f"--- Document {i} (Source: {source}, Page: {page}) ---\n"
                f"{doc.page_content}\n"
            )
        return "\n".join(context_parts)
    
    def _multi_stage_generate(
        self,
        prompt_template: PromptTemplate,
        prompt_inputs: Dict[str, Any],
        config: GenerationConfig,
        context: str,
        generation_trace: List[Dict[str, Any]]
    ) -> str:
        """
        Multi-stage generation with iterative refinement.
        
        Stage 1: Generate initial draft
        Stage 2+: Refine based on quality assessment
        """
        # Stage 1: Initial generation
        formatted_prompt = prompt_template.format(**prompt_inputs)
        response = self.llm.invoke(formatted_prompt)
        current_content = response.content if hasattr(response, 'content') else str(response)
        
        generation_trace.append({
            "stage": "generation",
            "iteration": 1,
            "type": "initial_draft"
        })
        
        # Iterative refinement
        for iteration in range(2, config.max_iterations + 1):
            # Assess current quality
            metrics = self.assess_quality(
                current_content,
                config.topic,
                config.document_type.value,
                context
            )
            
            # If quality is good enough, stop
            if metrics.overall_score >= 0.85:
                logger.info(f"Quality threshold met at iteration {iteration-1}")
                break
            
            # Generate refinement prompt
            refinement_prompt = f"""
You previously generated this document:

{current_content}

**QUALITY FEEDBACK**:
{chr(10).join(f"- {fb}" for fb in metrics.feedback)}

**SCORES**:
- Overall: {metrics.overall_score:.2f}
- Coherence: {metrics.coherence_score:.2f}
- Relevance: {metrics.relevance_score:.2f}
- Completeness: {metrics.completeness_score:.2f}

**TASK**: Improve the document by addressing the feedback above. Maintain the same structure and format, but enhance quality where needed.

**ORIGINAL CONTEXT**:
{context}

Generate the improved version:
"""
            
            response = self.llm.invoke(refinement_prompt)
            current_content = response.content if hasattr(response, 'content') else str(response)
            
            generation_trace.append({
                "stage": "generation",
                "iteration": iteration,
                "type": "refinement",
                "previous_score": metrics.overall_score
            })
        
        return current_content
    
    def generate_with_feedback(
        self,
        config: GenerationConfig,
        feedback: str,
        previous_generation: str,
        retrieved_docs: Optional[List[Document]] = None
    ) -> GeneratedDocument:
        """
        Generate refined document based on user feedback.
        
        Implements intelligent refinement that:
        - Preserves good sections
        - Focuses changes on feedback areas
        - Maintains document coherence
        """
        logger.info(f"Refining {config.document_type} based on feedback")
        
        # Retrieve context if not provided
        if retrieved_docs is None:
            retrieved_docs = self.adaptive_retrieve_context(
                config.topic,
                RetrievalStrategy(),
                config
            )
        
        # Analyze feedback to identify specific improvement areas
        feedback_analysis = self._analyze_feedback(feedback)
        
        # Format context
        if config.include_citations:
            context, doc_map = self.format_context_with_citations(retrieved_docs)
        else:
            context = self._format_context_simple(retrieved_docs)
            doc_map = {}
        
        # Create targeted refinement prompt
        refinement_prompt = f"""
You are refining a {config.document_type} document based on user feedback.

**ORIGINAL DOCUMENT**:
{previous_generation}

**USER FEEDBACK**:
{feedback}

**FEEDBACK ANALYSIS**:
{self._format_feedback_analysis(feedback_analysis)}

**SOURCE CONTEXT** (use to improve content):
{context}

**REFINEMENT STRATEGY**:
1. Identify sections that need improvement based on feedback
2. Preserve sections that are working well
3. Enhance content using additional context where needed
4. Maintain overall document structure and flow
5. Address all specific concerns raised in feedback

**REQUIREMENTS**:
- Keep the same document type and format
- Target audience: {config.target_audience}
- Language level: {config.language_level}
- Tone: {config.tone}

Generate the refined document:
"""
        
        response = self.llm.invoke(refinement_prompt)
        refined_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract citations
        citations = []
        if config.include_citations:
            citations = self.extract_citations(refined_content, doc_map)
        
        # Quality assessment
        quality_metrics = self.assess_quality(
            refined_content,
            config.topic,
            config.document_type.value,
            context
        )
        
        # Visual suggestions
        visual_suggestions = []
        if config.include_visual_suggestions:
            visual_suggestions = self.suggest_visuals(
                refined_content,
                config.document_type
            )
        
        # Prepare sources
        sources = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "snippet": doc.page_content[:200] + "..."
            }
            for doc in retrieved_docs
        ]
        
        result = GeneratedDocument(
            document_type=config.document_type.value,
            content=refined_content,
            metadata={
                "topic": config.topic,
                "persona": config.persona,
                "tone": config.tone,
                "length": config.length,
                "target_audience": config.target_audience,
                "language_level": config.language_level,
                "num_sources": len(retrieved_docs),
                "generation_method": "feedback_refinement",
                "feedback_incorporated": feedback
            },
            sources=sources,
            citations=citations,
            quality_metrics=quality_metrics,
            visual_suggestions=visual_suggestions,
            generation_trace=[
                {"stage": "feedback_analysis", "feedback_type": feedback_analysis},
                {"stage": "refinement", "quality_score": quality_metrics.overall_score}
            ]
        )
        
        logger.info(f"Refinement complete (Quality: {quality_metrics.overall_score:.2f})")
        return result
    
    def _analyze_feedback(self, feedback: str) -> Dict[str, Any]:
        """Analyze user feedback to identify improvement areas."""
        feedback_lower = feedback.lower()
        
        analysis = {
            "type": [],
            "focus_areas": [],
            "sentiment": "neutral"
        }
        
        # Identify feedback type
        if any(word in feedback_lower for word in ['too long', 'verbose', 'wordy']):
            analysis["type"].append("length_reduction")
            analysis["focus_areas"].append("brevity")
        
        if any(word in feedback_lower for word in ['too short', 'more detail', 'expand']):
            analysis["type"].append("length_expansion")
            analysis["focus_areas"].append("depth")
        
        if any(word in feedback_lower for word in ['unclear', 'confusing', 'hard to understand']):
            analysis["type"].append("clarity")
            analysis["focus_areas"].append("explanation")
        
        if any(word in feedback_lower for word in ['example', 'instance', 'illustration']):
            analysis["type"].append("examples_needed")
            analysis["focus_areas"].append("concrete_examples")
        
        if any(word in feedback_lower for word in ['tone', 'style', 'voice']):
            analysis["type"].append("tone_adjustment")
            analysis["focus_areas"].append("writing_style")
        
        if any(word in feedback_lower for word in ['structure', 'organize', 'flow']):
            analysis["type"].append("structure")
            analysis["focus_areas"].append("organization")
        
        if any(word in feedback_lower for word in ['missing', 'add', 'include', 'cover']):
            analysis["type"].append("content_addition")
            analysis["focus_areas"].append("completeness")
        
        # Sentiment
        if any(word in feedback_lower for word in ['great', 'good', 'excellent', 'but']):
            analysis["sentiment"] = "mostly_positive"
        elif any(word in feedback_lower for word in ['bad', 'poor', 'wrong', 'incorrect']):
            analysis["sentiment"] = "negative"
        
        return analysis
    
    def _format_feedback_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format feedback analysis for prompt."""
        if not analysis["type"]:
            return "General improvement requested."
        
        parts = []
        parts.append(f"Feedback type: {', '.join(analysis['type'])}")
        parts.append(f"Focus on: {', '.join(analysis['focus_areas'])}")
        parts.append(f"User sentiment: {analysis['sentiment']}")
        
        return "\n".join(parts)
    
    def generate_alternative_sections(
        self,
        config: GenerationConfig,
        section_name: str,
        current_content: str,
        retrieved_docs: List[Document],
        num_alternatives: int = 3
    ) -> Dict[str, str]:
        """
        Generate alternative versions of a specific section.
        Useful for A/B testing or giving users options.
        """
        logger.info(f"Generating {num_alternatives} alternatives for section: {section_name}")
        
        context = self._format_context_simple(retrieved_docs)
        
        alternatives = {}
        
        for i in range(num_alternatives):
            variation_prompt = f"""
Generate an alternative version of the "{section_name}" section.

**CURRENT VERSION**:
{current_content}

**CONTEXT**:
{context}

**REQUIREMENTS**:
- Variation {i+1}: {"More concise" if i == 0 else "More detailed with examples" if i == 1 else "Different angle/perspective"}
- Maintain factual accuracy
- Keep same tone: {config.tone}
- Target audience: {config.target_audience}

Generate the alternative version:
"""
            
            response = self.llm.invoke(variation_prompt)
            alternative_content = response.content if hasattr(response, 'content') else str(response)
            
            alternatives[f"alternative_{i+1}"] = alternative_content
        
        return alternatives
    
    def batch_generate(
        self,
        configs: List[GenerationConfig],
        shared_retrieval: bool = True
    ) -> List[GeneratedDocument]:
        """
        Generate multiple documents efficiently.
        
        Args:
            configs: List of generation configurations
            shared_retrieval: If True, retrieve context once for similar topics
        """
        logger.info(f"Batch generating {len(configs)} documents")
        
        results = []
        retrieval_cache = {}
        
        for config in configs:
            # Check if we can reuse retrieved context
            retrieved_docs = None
            if shared_retrieval and config.topic in retrieval_cache:
                retrieved_docs = retrieval_cache[config.topic]
                logger.info(f"Using cached retrieval for: {config.topic}")
            
            # Generate document
            result = self.generate_document(
                config,
                retrieved_docs=retrieved_docs
            )
            
            # Cache retrieval if enabled
            if shared_retrieval and config.topic not in retrieval_cache:
                # Retrieve fresh for caching
                docs = self.adaptive_retrieve_context(
                    config.topic,
                    RetrievalStrategy(),
                    config
                )
                retrieval_cache[config.topic] = docs
            
            results.append(result)
        
        logger.info(f"Batch generation complete: {len(results)} documents")
        return results
    
    def export_with_metadata(
        self,
        document: GeneratedDocument,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Export document with rich metadata in various formats.
        """
        if format == "json":
            import json
            return json.dumps(document.dict(), indent=2, default=str)
        
        elif format == "html":
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{document.metadata.get('topic', 'Document')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .quality-metrics {{ background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .citations {{ background: #fff3e0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .content {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="metadata">
        <h2>Document Metadata</h2>
        <p><strong>Type:</strong> {document.document_type}</p>
        <p><strong>Topic:</strong> {document.metadata.get('topic')}</p>
        <p><strong>Sources:</strong> {document.metadata.get('num_sources')}</p>
        <p><strong>Generated:</strong> {document.metadata.get('generation_method', 'standard')}</p>
    </div>
    
    {self._format_quality_metrics_html(document.quality_metrics) if document.quality_metrics else ''}
    
    <div class="content">
        {self._markdown_to_html(document.content)}
    </div>
    
    {self._format_citations_html(document.citations) if document.citations else ''}
    
    <div class="sources">
        <h3>Sources</h3>
        <ul>
        {''.join(f'<li>{src["source"]} (Page {src["page"]})</li>' for src in document.sources)}
        </ul>
    </div>
</body>
</html>
"""
            return html
        
        else:  # markdown
            md = f"""# {document.metadata.get('topic', 'Document')}

**Document Type:** {document.document_type}  
**Generated:** {document.metadata.get('generation_method', 'standard')}  
**Sources Used:** {document.metadata.get('num_sources')}

"""
            if document.quality_metrics:
                md += f"""## Quality Metrics
- Overall Score: {document.quality_metrics.overall_score:.2f}
- Coherence: {document.quality_metrics.coherence_score:.2f}
- Relevance: {document.quality_metrics.relevance_score:.2f}
- Completeness: {document.quality_metrics.completeness_score:.2f}

"""
            
            md += "---\n\n"
            md += document.content
            md += "\n\n---\n\n"
            
            md += "## Sources\n\n"
            for src in document.sources:
                md += f"- {src['source']} (Page {src['page']})\n"
            
            return md
    
    def _format_quality_metrics_html(self, metrics: QualityMetrics) -> str:
        """Format quality metrics as HTML."""
        return f"""
    <div class="quality-metrics">
        <h3>Quality Assessment</h3>
        <p><strong>Overall Score:</strong> {metrics.overall_score:.2f}</p>
        <p><strong>Coherence:</strong> {metrics.coherence_score:.2f}</p>
        <p><strong>Relevance:</strong> {metrics.relevance_score:.2f}</p>
        <p><strong>Completeness:</strong> {metrics.completeness_score:.2f}</p>
    </div>
"""
    
    def _format_citations_html(self, citations: List[CitationInfo]) -> str:
        """Format citations as HTML."""
        if not citations:
            return ""
        
        html = '<div class="citations"><h3>Citations</h3><ul>'
        for cit in citations:
            html += f'<li>{cit.doc_id}: {cit.source} (Page {cit.page})</li>'
        html += '</ul></div>'
        return html
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Basic markdown to HTML conversion."""
        # This is a simple implementation - use a proper markdown library in production
        html = markdown_text
        html = re.sub(r'^# (.+)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = html.replace('\n\n', '</p><p>')
        html = f'<p>{html}</p>'
        return html
