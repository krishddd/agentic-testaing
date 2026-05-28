"""
MCP Integration for Epistemic Agent

Provides real connectors for the epistemic agent:
- Web search via Tavily API
- Wikipedia fact checking
- File system operations
"""

import os
import glob
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SearchResultType(str, Enum):
    """Type of search result"""
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    LOCATION_NOT_FOUND = "location_not_found"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


@dataclass
class SearchResult:
    """Structured search result"""
    success: bool
    result_type: SearchResultType
    data: str
    raw_results: List[Dict[str, Any]] = None
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self):
        if self.raw_results is None:
            self.raw_results = []


class MCPSearchAdapter:
    """
    Adapter that connects the epistemic agent to real external tools.

    Provides:
    - Web search via Tavily API with semantic clustering
    - Wikipedia fact checking via LangChain
    - File system operations
    """

    def __init__(self):
        """Initialize the MCP adapter with real connectors."""
        self.tavily_client = None
        self._action_history: List[Dict[str, Any]] = []
        self._init_connectors()

    def _init_connectors(self):
        """Initialize real API connectors."""
        try:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=tavily_key)
                logger.info("Tavily search initialized")
            else:
                logger.warning("TAVILY_API_KEY not set — web search will return errors")
        except ImportError:
            logger.warning("Tavily package not installed — run: pip install tavily-python")
        except Exception as e:
            logger.error(f"Error initializing connectors: {e}")

    def web_search(self, query: str) -> SearchResult:
        """Perform a web search via Tavily with semantic clustering."""
        self._action_history.append({"action": "web_search", "query": query})

        # Guard against repeated identical searches
        recent_searches = [
            a for a in self._action_history[-5:]
            if a.get("action") == "web_search" and a.get("query") == query
        ]
        if len(recent_searches) > 2:
            return SearchResult(
                success=True,
                result_type=SearchResultType.NO_RESULTS,
                data="Search repeated multiple times without new information.",
                confidence=0.9, source="system"
            )

        if not self.tavily_client:
            return SearchResult(
                success=False,
                result_type=SearchResultType.ERROR,
                data="Web search unavailable: TAVILY_API_KEY not configured or tavily not installed.",
                source="system"
            )

        try:
            response = self.tavily_client.search(query=query, search_depth="basic")
            raw_results = response.get("results", [])

            if not raw_results:
                return SearchResult(
                    success=True,
                    result_type=SearchResultType.NO_RESULTS,
                    data=f"No results found for: {query}",
                    confidence=0.8, source="tavily"
                )

            # Semantic clustering for noise reduction
            clustered = self._semantic_cluster_results(raw_results)
            top_results = clustered[:3]

            formatted = []
            for r in top_results:
                formatted.append(f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:200]}")

            return SearchResult(
                success=True,
                result_type=SearchResultType.SUCCESS,
                data="\n".join(formatted),
                raw_results=top_results,
                confidence=0.75, source="tavily"
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return SearchResult(
                success=False,
                result_type=SearchResultType.ERROR,
                data=f"Search error: {str(e)}",
                source="tavily"
            )

    def _semantic_cluster_results(self, results: List[Dict]) -> List[Dict]:
        """
        Groups semantically similar results and filters noise.
        Falls back to returning results as-is if sklearn unavailable.
        """
        if len(results) < 3:
            return results

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            import numpy as np

            corpus = [f"{r.get('title', '')} {r.get('content', '')}" for r in results]
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(corpus)

            n_clusters = max(2, len(results) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(X)

            labels = kmeans.labels_
            counts = np.bincount(labels)
            dense_clusters = np.argsort(counts)[::-1]

            filtered = []
            for cluster_idx in dense_clusters:
                indices = np.where(labels == cluster_idx)[0]
                for idx in indices:
                    filtered.append(results[idx])

            return filtered

        except ImportError:
            logger.debug("sklearn not available, skipping clustering")
            return results
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return results

    def list_files(self, path: str = ".", pattern: str = "*") -> SearchResult:
        """List files in a directory."""
        self._action_history.append({"action": "list_files", "path": path, "pattern": pattern})

        try:
            files = glob.glob(os.path.join(path, pattern))

            if not files:
                return SearchResult(
                    success=True,
                    result_type=SearchResultType.NO_RESULTS,
                    data=f"No files found matching '{pattern}' in '{path}'",
                    confidence=0.9, source="filesystem"
                )

            if len(files) > 1:
                base_names = [os.path.basename(f) for f in files]
                common_prefix = os.path.commonprefix(base_names)
                if len(common_prefix) > 3:
                    return SearchResult(
                        success=True,
                        result_type=SearchResultType.AMBIGUOUS,
                        data=f"Found multiple similar files: {base_names}. Clarification needed.",
                        raw_results=[{"filename": f} for f in files],
                        confidence=0.85, source="filesystem"
                    )

            return SearchResult(
                success=True,
                result_type=SearchResultType.SUCCESS,
                data=f"Files found: {files}",
                raw_results=[{"filename": f} for f in files],
                confidence=0.95, source="filesystem"
            )

        except Exception as e:
            return SearchResult(
                success=False,
                result_type=SearchResultType.ERROR,
                data=f"Error listing files: {str(e)}",
                source="filesystem"
            )

    def read_file(self, filepath: str) -> SearchResult:
        """Read contents of a file."""
        self._action_history.append({"action": "read_file", "filepath": filepath})

        try:
            if not os.path.exists(filepath):
                return SearchResult(
                    success=False,
                    result_type=SearchResultType.NO_RESULTS,
                    data=f"File not found: {filepath}",
                    confidence=0.95, source="filesystem"
                )

            if not os.path.isfile(filepath):
                return SearchResult(
                    success=False,
                    result_type=SearchResultType.ERROR,
                    data=f"Path is not a file: {filepath}",
                    source="filesystem"
                )

            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(500)  # Read first 500 chars

            return SearchResult(
                success=True,
                result_type=SearchResultType.SUCCESS,
                data=f"Contents of {os.path.basename(filepath)}:\n{content}",
                confidence=0.95, source="filesystem"
            )

        except Exception as e:
            return SearchResult(
                success=False,
                result_type=SearchResultType.ERROR,
                data=f"Error reading file: {str(e)}",
                source="filesystem"
            )

    def wikipedia_search(self, query: str) -> SearchResult:
        """Search Wikipedia for fact verification."""
        self._action_history.append({"action": "wikipedia", "query": query})

        try:
            from langchain_community.utilities import WikipediaAPIWrapper
            wiki = WikipediaAPIWrapper()
            result = wiki.run(query)

            if not result or "No good Wikipedia Search Result was found" in result:
                return SearchResult(
                    success=True,
                    result_type=SearchResultType.NO_RESULTS,
                    data=f"No Wikipedia article found for: {query}",
                    confidence=0.8, source="wikipedia"
                )

            return SearchResult(
                success=True,
                result_type=SearchResultType.SUCCESS,
                data=result[:500],
                confidence=0.85, source="wikipedia"
            )

        except ImportError:
            return SearchResult(
                success=False,
                result_type=SearchResultType.ERROR,
                data="Wikipedia search unavailable: langchain_community not installed.",
                source="wikipedia"
            )
        except Exception as e:
            return SearchResult(
                success=False,
                result_type=SearchResultType.ERROR,
                data=f"Wikipedia search error: {str(e)}",
                source="wikipedia"
            )

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the history of actions taken."""
        return self._action_history.copy()

    def clear_history(self):
        """Clear action history."""
        self._action_history = []

    def has_repeated_action(self, action_name: str, max_repeats: int = 2) -> bool:
        """Check if an action has been repeated too many times."""
        recent = self._action_history[-5:]
        count = sum(1 for a in recent if a.get("action") == action_name)
        return count >= max_repeats
