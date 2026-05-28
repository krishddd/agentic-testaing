import json
import logging
from typing import List, Optional, Any, Dict, Tuple, Set
from pydantic import BaseModel, Field
from langchain_core.language_models.base import BaseLanguageModel
import re
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR MIND MAP STRUCTURE
# ============================================================================

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class MindMapNode(BaseModel):
    """Enhanced node in the mind map with rich metadata."""
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="The text label displayed on the node")
    level: int = Field(..., description="Hierarchy level (0 = root, 1 = main branch, etc.)")
    node_type: str = Field(default="concept", description="Type: root, concept, detail, insight, example")
    importance: int = Field(default=5, ge=1, le=10, description="Importance score (1-10)")
    color: Optional[str] = Field(None, description="Suggested color for visualization")
    icon: Optional[str] = Field(None, description="Suggested icon/emoji")
    description: Optional[str] = Field(None, description="Additional context for the node")
    keywords: List[str] = Field(default_factory=list, description="Related keywords")
    
    model_config = ConfigDict(
        populate_by_name=True,  # Allow using field name or alias
        extra='ignore'  # Ignore extra fields from LLM
    )

class MindMapEdge(BaseModel):
    """Enhanced connection between nodes with relationship metadata."""
    from_node: str = Field(..., alias="from", description="The ID of the source node")
    to_node: str = Field(..., alias="to", description="The ID of the target node")
    relationship_type: str = Field(default="contains", description="Type: contains, leads_to, supports, contrasts")
    weight: float = Field(default=1.0, ge=0.1, le=10.0, description="Edge weight/strength")
    label: Optional[str] = Field(None, description="Optional label for the edge")
    
    model_config = ConfigDict(
        populate_by_name=True,  # KEY FIX: Allow both 'from'/'to' and 'from_node'/'to_node'
        extra='ignore'
    )

class MindMapCluster(BaseModel):
    """Represents a thematic cluster of related nodes."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    theme: str = Field(..., description="The common theme/topic")
    node_ids: List[str] = Field(..., description="IDs of nodes in this cluster")
    color: str = Field(..., description="Color for the cluster")
    
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

class MindMapStatistics(BaseModel):
    """Statistics about the generated mind map."""
    total_nodes: int
    total_edges: int
    max_depth: int
    avg_branching_factor: float
    node_distribution: Dict[int, int]
    cluster_count: int
    generation_time: float
    
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

class MindMapResponse(BaseModel):
    """The complete enhanced mind map structure."""
    root_node_id: str = Field(..., description="The ID of the central root node")
    nodes: List[MindMapNode] = Field(..., description="List of all nodes in the map")
    edges: List[MindMapEdge] = Field(..., description="List of all edges connecting the nodes")
    clusters: List[MindMapCluster] = Field(default_factory=list, description="Thematic clusters")
    statistics: Optional[MindMapStatistics] = Field(None, description="Map statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    layout_suggestions: Dict[str, Any] = Field(default_factory=dict, description="Visualization hints")
    
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

# ============================================================================
# ADVANCED MIND MAP GENERATOR
# ============================================================================

class AdvancedMindMapGenerator:
    """
    Advanced mind map generator with:
    - Multi-level hierarchy (up to 5 levels deep)
    - Intelligent node clustering
    - Relationship typing
    - Importance scoring
    - Color coding
    - Layout optimization
    """
    
    # Color palette for different node types and levels
    COLOR_PALETTE = {
        0: "#FF6B6B",  # Root - Red
        1: "#4ECDC4",  # Level 1 - Teal
        2: "#45B7D1",  # Level 2 - Blue
        3: "#96CEB4",  # Level 3 - Green
        4: "#FFEAA7",  # Level 4 - Yellow
        5: "#DFE6E9",  # Level 5 - Gray
    }
    
    NODE_TYPE_COLORS = {
        "root": "#FF6B6B",
        "concept": "#4ECDC4",
        "detail": "#45B7D1",
        "insight": "#F39C12",
        "example": "#9B59B6",
        "question": "#E74C3C"
    }
    
    RELATIONSHIP_TYPES = [
        "contains", "leads_to", "supports", "contrasts", 
        "defines", "exemplifies", "elaborates", "connects_to"
    ]
    
    def __init__(self, llm: BaseLanguageModel, max_nodes: int = 50, max_depth: int = 5):
        """
        Initialize the advanced mind map generator.
        
        Args:
            llm: Language model for generation
            max_nodes: Maximum number of nodes to generate
            max_depth: Maximum depth of the hierarchy
        """
        self.llm = llm
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.generation_start_time = None

    def _create_generation_prompt(self, text_content: str, topic: str) -> str:
        """
        Create an enhanced prompt with strict JSON output requirements.
        """
        # Calculate target nodes based on max_nodes
        target_min = min(15, self.max_nodes // 2)
        target_max = min(self.max_nodes, 50)
        
        return f"""You are a JSON generator for mind maps. Output ONLY valid JSON, no explanations.
    
    TOPIC: {topic}
    
    TEXT TO ANALYZE:
    {text_content[:20000]}  # Limit to prevent context overflow
    
    GENERATE A HIERARCHICAL MIND MAP (Max {self.max_depth} levels, {target_min}-{target_max} nodes)
    
    CRITICAL RULES:
    1. Output ONLY the JSON object - NO markdown, NO explanations, NO extra text
    2. Ensure ALL strings are properly closed with double quotes
    3. Ensure ALL arrays and objects are properly closed
    4. NO trailing commas before closing braces/brackets
    5. Use ONLY ASCII characters in labels (avoid special unicode)
    
    JSON STRUCTURE (EXACT FORMAT):
    {{
      "root_node_id": "node_0",
      "nodes": [
        {{
          "id": "node_0",
          "label": "{topic}",
          "level": 0,
          "node_type": "root",
          "importance": 10,
          "description": "Root topic description",
          "keywords": ["key1", "key2"]
        }},
        {{
          "id": "node_1",
          "label": "Main Concept 1",
          "level": 1,
          "node_type": "concept",
          "importance": 8,
          "description": "Description here",
          "keywords": ["key1", "key2"]
        }}
      ],
      "edges": [
        {{
          "from": "node_0",
          "to": "node_1",
          "relationship_type": "contains",
          "weight": 1.0,
          "label": null
        }}
      ]
    }}
    
    NODE TYPES: root, concept, detail, insight, example
    RELATIONSHIP TYPES: contains, leads_to, supports, contrasts, defines
    IMPORTANCE: 1-10 (10=critical, 5=relevant, 1=minor)
    
    REQUIREMENTS:
    - Generate {target_min} to {target_max} nodes (including root)
    - Create a balanced hierarchy (not too linear, not too bushy)
    - Every non-root node MUST have a parent edge
    - Use concise labels (3-8 words max)
    - Keep descriptions under 100 characters
    - Ensure valid JSON - check all quotes and brackets
    
    START JSON OUTPUT NOW (no preamble):"""

    def _parse_llm_output(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """
        Ultra-robust JSON parsing with multiple fallback strategies.
        """
        
        # Strategy 1: Direct parse
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError as e1:
            logger.warning(f"Direct JSON parse failed: {e1}")
        
        # Strategy 2: Extract from ```json ... ```
        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', llm_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.warning("JSON block extraction failed")
        
        # Strategy 3: Find first { to last }
        json_match = re.search(r'\{[\s\S]*\}', llm_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Brace extraction failed")
        
        # Strategy 4: Clean common issues
        cleaned = llm_output
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)  # Remove trailing commas
        cleaned = re.sub(r'//.*?\n', '\n', cleaned)  # Remove // comments
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)  # Remove /* */ comments
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Cleaned JSON parse failed")
        
        # Strategy 5: Fix unterminated strings
        try:
            # Find the last valid closing brace
            last_brace = llm_output.rfind('}')
            if last_brace > 0:
                truncated = llm_output[:last_brace + 1]
                return json.loads(truncated)
        except json.JSONDecodeError:
            logger.warning("Truncation strategy failed")
        
        # Strategy 6: Aggressive fix - find and complete the JSON
        try:
            # Extract up to "nodes" array and try to salvage
            nodes_match = re.search(r'"nodes"\s*:\s*\[([\s\S]*?)(?:\]|$)', llm_output)
            edges_match = re.search(r'"edges"\s*:\s*\[([\s\S]*?)(?:\]|$)', llm_output)
            root_match = re.search(r'"root_node_id"\s*:\s*"([^"]+)"', llm_output)
            
            if nodes_match and root_match:
                # Try to build valid JSON from fragments
                nodes_str = nodes_match.group(1)
                
                # Remove incomplete last object
                nodes_str = re.sub(r',\s*\{[^}]*$', '', nodes_str)
                
                # Build minimal valid structure
                reconstructed = {
                    "root_node_id": root_match.group(1),
                    "nodes": [],
                    "edges": []
                }
                
                # Try to parse nodes
                try:
                    nodes_json = json.loads(f'[{nodes_str}]')
                    reconstructed["nodes"] = nodes_json
                except:
                    pass
                
                # Try to parse edges if found
                if edges_match:
                    try:
                        edges_str = edges_match.group(1)
                        edges_str = re.sub(r',\s*\{[^}]*$', '', edges_str)
                        edges_json = json.loads(f'[{edges_str}]')
                        reconstructed["edges"] = edges_json
                    except:
                        pass
                
                if reconstructed["nodes"]:
                    logger.info(f"Reconstructed JSON with {len(reconstructed['nodes'])} nodes")
                    return reconstructed
        except Exception as e:
            logger.warning(f"Reconstruction strategy failed: {e}")
        
        # Strategy 7: Last resort - extract any valid JSON objects
        try:
            # Find all complete node objects
            node_pattern = r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"label"\s*:\s*"[^"]*"\s*,.*?\}'
            found_nodes = re.findall(node_pattern, llm_output, re.DOTALL)
            
            if found_nodes:
                nodes = []
                for node_str in found_nodes[:20]:  # Limit to 20 nodes
                    try:
                        node = json.loads(node_str)
                        nodes.append(node)
                    except:
                        continue
                
                if nodes:
                    logger.info(f"Extracted {len(nodes)} valid node objects")
                    
                    # Find root node
                    root_id = "node_0"
                    for node in nodes:
                        if node.get("level") == 0:
                            root_id = node.get("id")
                            break
                    
                    # Build basic edges (each node connects to root or previous)
                    edges = []
                    for i, node in enumerate(nodes[1:], 1):
                        edges.append({
                            "from": root_id if i == 1 else f"node_{i-1}",
                            "to": node.get("id"),
                            "relationship_type": "contains",
                            "weight": 1.0
                        })
                    
                    return {
                        "root_node_id": root_id,
                        "nodes": nodes,
                        "edges": edges
                    }
        except Exception as e:
            logger.warning(f"Object extraction failed: {e}")
        
        # All strategies failed
        logger.error("All JSON parsing strategies failed")
        logger.error(f"Raw output sample (first 2000 chars):\n{llm_output[:2000]}")
        logger.error(f"Raw output sample (last 500 chars):\n{llm_output[-500:]}")
        return None

    def _validate_and_enhance_structure(self, raw_data: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """
        Validate the structure and add missing elements.
        """
        # Ensure root node exists
        if "root_node_id" not in raw_data:
            raw_data["root_node_id"] = "node_0"
        
        # Ensure nodes list exists
        if "nodes" not in raw_data or not raw_data["nodes"]:
            logger.warning("No nodes found. Creating minimal structure.")
            raw_data["nodes"] = [{
                "id": "node_0",
                "label": topic,
                "level": 0,
                "node_type": "root",
                "importance": 10
            }]
        
        # Ensure edges list exists
        if "edges" not in raw_data:
            raw_data["edges"] = []
        
        # Add colors based on level
        for node in raw_data["nodes"]:
            if "color" not in node or not node["color"]:
                level = node.get("level", 0)
                node_type = node.get("node_type", "concept")
                node["color"] = self.NODE_TYPE_COLORS.get(node_type, self.COLOR_PALETTE.get(level, "#95A5A6"))
        
        # Normalize edges to use from_node/to_node
        normalized_edges = []
        for edge in raw_data["edges"]:
            normalized_edges.append({
                "from_node": edge.get("from", edge.get("from_node")),
                "to_node": edge.get("to", edge.get("to_node")),
                "relationship_type": edge.get("relationship_type", "contains"),
                "weight": edge.get("weight", 1.0),
                "label": edge.get("label")
            })
        raw_data["edges"] = normalized_edges
        
        return raw_data

    def _detect_clusters(self, nodes: List[MindMapNode], edges: List[MindMapEdge]) -> List[MindMapCluster]:
        """
        Detect thematic clusters in the mind map using connected components at level 1.
        """
        clusters = []
        
        # Group level 1 nodes (main branches)
        level_1_nodes = [n for n in nodes if n.level == 1]
        
        # Create clusters based on level 1 branches
        cluster_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DFE6E9"]
        
        for idx, l1_node in enumerate(level_1_nodes):
            # Find all descendants of this level 1 node
            descendants = self._find_descendants(l1_node.id, edges, nodes)
            
            if descendants:
                cluster = MindMapCluster(
                    cluster_id=f"cluster_{idx}",
                    theme=l1_node.label,
                    node_ids=[l1_node.id] + descendants,
                    color=cluster_colors[idx % len(cluster_colors)]
                )
                clusters.append(cluster)
        
        return clusters

    def _find_descendants(self, node_id: str, edges: List[MindMapEdge], nodes: List[MindMapNode]) -> List[str]:
        """
        Find all descendants of a given node.
        """
        descendants = []
        to_explore = [node_id]
        explored = set()
        
        while to_explore:
            current = to_explore.pop(0)
            if current in explored:
                continue
            explored.add(current)
            
            # Find children
            for edge in edges:
                if edge.from_node == current and edge.to_node not in explored:
                    descendants.append(edge.to_node)
                    to_explore.append(edge.to_node)
        
        return descendants

    def _calculate_statistics(self, nodes: List[MindMapNode], edges: List[MindMapEdge], clusters: List[MindMapCluster]) -> MindMapStatistics:
        """
        Calculate comprehensive statistics about the mind map.
        """
        # Node distribution by level
        node_distribution = defaultdict(int)
        max_depth = 0
        for node in nodes:
            node_distribution[node.level] += 1
            max_depth = max(max_depth, node.level)
        
        # Average branching factor
        parent_children_count = defaultdict(int)
        for edge in edges:
            parent_children_count[edge.from_node] += 1
        
        avg_branching = (
            sum(parent_children_count.values()) / len(parent_children_count)
            if parent_children_count else 0
        )
        
        generation_time = 0
        if self.generation_start_time:
            generation_time = datetime.now().timestamp() - self.generation_start_time
        
        return MindMapStatistics(
            total_nodes=len(nodes),
            total_edges=len(edges),
            max_depth=max_depth,
            avg_branching_factor=round(avg_branching, 2),
            node_distribution=dict(node_distribution),
            cluster_count=len(clusters),
            generation_time=round(generation_time, 2)
        )

    def _create_layout_suggestions(self, nodes: List[MindMapNode], statistics: MindMapStatistics) -> Dict[str, Any]:
        """
        Create layout optimization suggestions for visualization.
        """
        return {
            "recommended_layout": "radial" if statistics.total_nodes < 30 else "hierarchical",
            "node_spacing": {
                "horizontal": 150 if statistics.avg_branching_factor > 4 else 200,
                "vertical": 100
            },
            "canvas_size": {
                "width": max(1200, statistics.total_nodes * 40),
                "height": max(800, statistics.max_depth * 200)
            },
            "zoom_level": 1.0 if statistics.total_nodes < 40 else 0.8,
            "show_edge_labels": statistics.total_edges < 50,
            "cluster_visualization": True if statistics.cluster_count > 1 else False
        }

    def generate_from_text(
        self, 
        text_content: str, 
        topic: str,
        focus_areas: Optional[List[str]] = None,
        depth_preference: str = "balanced"
    ) -> MindMapResponse:
        """
        Generate an advanced mind map from text content with robust error handling.
        """
        self.generation_start_time = datetime.now().timestamp()
        
        if not text_content or not text_content.strip():
            raise ValueError("Input text content cannot be empty.")
        
        # Adjust max_depth based on preference
        depth_map = {"shallow": 3, "balanced": 5, "deep": 7}
        self.max_depth = depth_map.get(depth_preference, 5)
        
        # Intelligent text truncation
        max_chars = 20000  # Reduced for better LLM handling
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars]
            last_period = text_content.rfind('.')
            if last_period > max_chars * 0.8:
                text_content = text_content[:last_period + 1]
            logger.info(f"Truncated text to {len(text_content)} characters")
        
        # Add focus areas if provided
        if focus_areas:
            focus_instruction = f"\n\nFOCUS ON: {', '.join(focus_areas)}"
            text_content = focus_instruction + "\n\n" + text_content
        
        # Generate prompt
        prompt = self._create_generation_prompt(text_content, topic)
        
        logger.info(f"Generating mind map: topic='{topic}', max_nodes={self.max_nodes}, depth={self.max_depth}")
        
        try:
            # Invoke LLM with retries
            max_retries = 2
            raw_data = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"LLM invocation attempt {attempt + 1}/{max_retries}")
                    
                    response = self.llm.invoke(prompt)
                    llm_output = response.content if hasattr(response, 'content') else str(response)
                    
                    logger.info(f"LLM response length: {len(llm_output)} characters")
                    logger.debug(f"LLM output preview: {llm_output[:500]}...")
                    
                    # Parse JSON
                    raw_data = self._parse_llm_output(llm_output)
                    
                    if raw_data:
                        logger.info("Successfully parsed LLM output")
                        break
                    else:
                        last_error = "Failed to parse LLM output as valid JSON"
                        if attempt < max_retries - 1:
                            logger.warning(f"Parse failed, retrying with simplified prompt...")
                            # Simplify for retry
                            self.max_nodes = max(10, self.max_nodes // 2)
                            prompt = self._create_generation_prompt(text_content[:10000], topic)
                        
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
            
            # Check if we got valid data
            if not raw_data:
                logger.error("All parsing attempts failed")
                raise ValueError(last_error or "Failed to generate valid mind map structure")
            
            # Validate and enhance
            enhanced_data = self._validate_and_enhance_structure(raw_data, topic)
            
            # Ensure minimum structure
            if len(enhanced_data.get("nodes", [])) < 2:
                logger.warning("Insufficient nodes generated, creating minimal structure")
                enhanced_data = self._create_minimal_structure(topic)
            
            # Parse with Pydantic
            nodes = [MindMapNode(**node_data) for node_data in enhanced_data["nodes"]]
            edges = [MindMapEdge(**{
                "from": edge_data.get("from", edge_data.get("from_node")),
                "to": edge_data.get("to", edge_data.get("to_node")),
                "relationship_type": edge_data.get("relationship_type", "contains"),
                "weight": edge_data.get("weight", 1.0),
                "label": edge_data.get("label")
            }) for edge_data in enhanced_data["edges"]]
            
            logger.info(f"Parsed {len(nodes)} nodes and {len(edges)} edges")
            
            # Detect clusters
            clusters = self._detect_clusters(nodes, edges)
            
            # Calculate statistics
            statistics = self._calculate_statistics(nodes, edges, clusters)
            
            # Create layout suggestions
            layout_suggestions = self._create_layout_suggestions(nodes, statistics)
            
            # Build response
            mind_map = MindMapResponse(
                root_node_id=enhanced_data["root_node_id"],
                nodes=nodes,
                edges=edges,
                clusters=clusters,
                statistics=statistics,
                layout_suggestions=layout_suggestions,
                metadata={
                    "topic": topic,
                    "generation_timestamp": datetime.now().isoformat(),
                    "source_text_length": len(text_content),
                    "focus_areas": focus_areas,
                    "depth_preference": depth_preference,
                    "max_nodes_limit": self.max_nodes,
                    "max_depth_limit": self.max_depth,
                    "generation_attempts": attempt + 1
                }
            )
            
            logger.info(f"✓ Mind map generated successfully!")
            logger.info(f"  Nodes: {statistics.total_nodes}, Edges: {statistics.total_edges}")
            logger.info(f"  Depth: {statistics.max_depth}, Clusters: {statistics.cluster_count}")
            logger.info(f"  Generation time: {statistics.generation_time}s")
            
            return mind_map
            
        except Exception as e:
            logger.error(f"Error generating mind map: {e}", exc_info=True)
            return self._create_fallback_mindmap(topic, str(e))
    
    
    def _create_minimal_structure(self, topic: str) -> Dict[str, Any]:
        """Create a minimal but valid mind map structure."""
        return {
            "root_node_id": "node_0",
            "nodes": [
                {
                    "id": "node_0",
                    "label": topic,
                    "level": 0,
                    "node_type": "root",
                    "importance": 10,
                    "description": "Central topic",
                    "keywords": [topic.split()[0] if topic.split() else "topic"]
                },
                {
                    "id": "node_1",
                    "label": "Key Concept",
                    "level": 1,
                    "node_type": "concept",
                    "importance": 7,
                    "description": "Main concept",
                    "keywords": ["concept"]
                }
            ],
            "edges": [
                {
                    "from": "node_0",
                    "to": "node_1",
                    "relationship_type": "contains",
                    "weight": 1.0,
                    "label": None
                }
            ]
        }

    def _create_fallback_mindmap(self, topic: str, error: str) -> MindMapResponse:
        """
        Create a minimal fallback mind map when generation fails.
        """
        fallback_node = MindMapNode(
            id="node_0",
            label=topic,
            level=0,
            node_type="root",
            importance=10,
            color=self.COLOR_PALETTE[0],
            description=f"Generation failed: {error[:100]}"
        )
        
        error_node = MindMapNode(
            id="node_error",
            label=f"Error: {error[:50]}",
            level=1,
            node_type="detail",
            importance=1,
            color="#E74C3C"
        )
        
        # FIX: Use the aliased field names 'from' and 'to' instead of 'from_node' and 'to_node'
        fallback_edge = MindMapEdge(
            **{
                "from": "node_0",  # Use alias
                "to": "node_error",  # Use alias
                "relationship_type": "contains",
                "weight": 1.0
            }
        )
        
        return MindMapResponse(
            root_node_id="node_0",
            nodes=[fallback_node, error_node],
            edges=[fallback_edge],
            clusters=[],
            statistics=MindMapStatistics(
                total_nodes=2,
                total_edges=1,
                max_depth=1,
                avg_branching_factor=1.0,
                node_distribution={0: 1, 1: 1},
                cluster_count=0,
                generation_time=0.0
            ),
            metadata={"error": error, "fallback": True}
        )

    def export_to_json_file(self, mind_map: MindMapResponse, output_path: str) -> str:
        """
        Export the mind map to a JSON file.
        
        Args:
            mind_map: The MindMapResponse object to export
            output_path: Path where to save the JSON file
            
        Returns:
            The actual path where the file was saved
        """
        try:
            # Convert to dict using Pydantic's model_dump
            data = mind_map.model_dump(by_alias=True)
            
            # Write to file with pretty formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Mind map exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export mind map: {e}", exc_info=True)
            raise

    def export_to_mermaid(self, mind_map: MindMapResponse) -> str:
        """
        Export mind map to Mermaid diagram format for visualization.
        
        Returns:
            Mermaid markdown string
        """
        lines = ["graph TD"]
        
        # Add nodes with styling
        for node in mind_map.nodes:
            style = "(" if node.level == 0 else "["
            end_style = ")" if node.level == 0 else "]"
            
            # Escape special characters
            label = node.label.replace('"', "'")
            lines.append(f'    {node.id}{style}"{label}"{end_style}')
        
        # Add edges
        for edge in mind_map.edges:
            arrow = "==>" if edge.weight > 1.5 else "-->"
            label_text = f'|{edge.label}|' if edge.label else ''
            lines.append(f'    {edge.from_node} {arrow}{label_text} {edge.to_node}')
        
        # Add styling
        for node in mind_map.nodes:
            if node.color:
                color_hex = node.color.replace('#', '')
                lines.append(f'    style {node.id} fill:{node.color},stroke:#333,stroke-width:2px')
        
        return '\n'.join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_mindmap_from_text(
    llm: BaseLanguageModel,
    text: str,
    topic: str,
    output_path: Optional[str] = None,
    max_nodes: int = 50,
    depth: str = "balanced"
) -> Tuple[MindMapResponse, Optional[str]]:
    """
    Convenience function to generate a mind map and optionally save it.
    
    Args:
        llm: Language model
        text: Source text
        topic: Central topic
        output_path: Optional path to save JSON
        max_nodes: Maximum nodes to generate
        depth: Depth preference ("shallow", "balanced", "deep")
        
    Returns:
        Tuple of (MindMapResponse, saved_file_path)
    """
    generator = AdvancedMindMapGenerator(llm, max_nodes=max_nodes)
    mind_map = generator.generate_from_text(text, topic, depth_preference=depth)
    
    saved_path = None
    if output_path:
        saved_path = generator.export_to_json_file(mind_map, output_path)
    
    return mind_map, saved_path


# ============================================================================
# EXAMPLE USAGE (for testing)
# ============================================================================

if __name__ == "__main__":
    # This is just for demonstration - you'd integrate with your actual LLM
    print("Advanced Mind Map Generator Module")
    print("=" * 50)
    print("\nFeatures:")
    print("  ✓ Multi-level hierarchies (up to 7 levels)")
    print("  ✓ Intelligent clustering")
    print("  ✓ Relationship typing")
    print("  ✓ Importance scoring")
    print("  ✓ Color coding")
    print("  ✓ Layout optimization")
    print("  ✓ Statistics and metadata")
    print("  ✓ Export to JSON and Mermaid")
    print("\nReady for integration!")
