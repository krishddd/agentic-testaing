
import logging
import networkx as nx
import pickle
import os
import json
from typing import List, Tuple, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

class GraphKnowledgeManager:
    """
    Manages the Knowledge Graph using NetworkX.
    Extracts entities and relations from text, builds the graph, and queries it.
    """

    def __init__(self, llm: BaseChatModel, graph_path: str = "results/knowledge_graph.pkl"):
        self.llm = llm
        self.graph_path = graph_path
        self.graph = nx.Graph()
        self._load_graph()

    def _load_graph(self):
        """Loads the graph from disk if it exists."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
        else:
            logger.info("Initializing new empty knowledge graph.")

    def save_graph(self):
        """Saves the graph to disk."""
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        try:
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Saved knowledge graph with {self.graph.number_of_nodes()} nodes.")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """
        Uses LLM to extract (Subject, Relation, Object) triplets from text.
        """
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract knowledge triplets from the following text.
Each triplet should consist of a Subject (Entity), a Relation, and an Object (Entity).
Focus on relationships between people, organizations, locations, and key concepts.
Ignore generic or vague statements.

Text: {text}

Return ONLY a JSON object with a single key "triplets" containing a list of objects like:
{{"triplets": [{{"subject": "Elon Musk", "relation": "CEO of", "object": "Tesla"}}]}}
"""
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            # Chunk text if too large (simplified here)
            result = chain.invoke({"text": text[:4000]})
            return result.get("triplets", [])
        except Exception as e:
            logger.error(f"Triplet extraction failed: {e}")
            return []

    def build_graph_from_documents(self, documents: List[str]):
        """
        Iterates over documents, extracts triplets, and updates the graph.
        """
        logger.info(f"Building graph from {len(documents)} documents...")
        count = 0
        for doc in documents:
            triplets = self.extract_triplets(doc)
            for triplet in triplets:
                subj = triplet.get("subject")
                obj_ = triplet.get("object")
                rel = triplet.get("relation")
                
                if subj and obj_ and rel:
                    self.graph.add_node(subj, type="entity")
                    self.graph.add_node(obj_, type="entity")
                    
                    # Add edge with relation as attribute, store list of relations if multiple exist
                    if self.graph.has_edge(subj, obj_):
                        existing_rels = self.graph[subj][obj_].get('relations', [])
                        if rel not in existing_rels:
                            existing_rels.append(rel)
                            self.graph[subj][obj_]['relations'] = existing_rels
                    else:
                        self.graph.add_edge(subj, obj_, relations=[rel])
            count += 1
            if count % 10 == 0:
                logger.info(f"Processed {count}/{len(documents)} documents for graph.")
        
        self.save_graph()

    def get_context_for_query(self, query: str, hops: int = 1) -> str:
        """
        1. Extract entities from the query.
        2. Find them in the graph.
        3. Traverse 'hops' degrees.
        4. Return a text summary of connections.
        """
        # 1. Extract entities from query using LLM
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Extract the key entities (Subject/Object) from this query. Return JSON list ['Entity1', 'Entity2']. Query: {query}"
        )
        try:
            chain = prompt | self.llm | JsonOutputParser()
            query_entities = chain.invoke({"query": query})
            if isinstance(query_entities, dict):
                 query_entities = query_entities.get('entities', [])
        except Exception:
            return ""

        context_lines = []
        found_entities = []

        # 2. Find in graph
        for entity in query_entities:
            # Simple fuzzy match or direct lookup
            # For this MVP, we try direct match and case-insensitive match
            match = None
            if entity in self.graph:
                match = entity
            else:
                # Naive case insensitive search
                for node in self.graph.nodes():
                    if node.lower() == entity.lower():
                        match = node
                        break
            
            if match:
                found_entities.append(match)
                # 3. Traverse
                subgraph = nx.ego_graph(self.graph, match, radius=hops)
                
                for u, v, data in subgraph.edges(data=True):
                    rels = ", ".join(data.get('relations', []))
                    context_lines.append(f"{u} --[{rels}]--> {v}")

        if not context_lines:
            return ""

        # Dedup
        context_lines = list(set(context_lines))
        return "Graph Knowledge Context:\n" + "\n".join(context_lines)
