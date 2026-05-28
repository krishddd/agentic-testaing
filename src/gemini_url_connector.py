"""
Enhanced Gemini URL Connector with News Site Intelligence
Handles news aggregators and extracts multiple sub-articles automatically
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
import re
from urllib.parse import urljoin, urlparse

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI library not installed. Install with: pip install google-genai")

logger = logging.getLogger(__name__)

@dataclass
class ArticleInfo:
    """Information about a single article"""
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class GeminiURLResponse:
    """Enhanced response with support for multiple articles"""
    success: bool
    content: Optional[str] = None
    urls_analyzed: List[str] = field(default_factory=list)
    articles: List[ArticleInfo] = field(default_factory=list)
    error: Optional[str] = None
    grounding_metadata: Optional[Dict] = None
    is_news_site: bool = False
    total_articles_found: int = 0

class NewsPatterns:
    """Patterns to detect and handle news websites"""
    
    NEWS_DOMAINS = [
        'news.google.com', 'bbc.com', 'cnn.com', 'reuters.com',
        'nytimes.com', 'theguardian.com', 'aljazeera.com',
        'apnews.com', 'bloomberg.com', 'forbes.com', 'techcrunch.com',
        'verge.com', 'wired.com', 'medium.com', 'ndtv.com',
        'timesofindia.indiatimes.com', 'hindustantimes.com',
        'indianexpress.com', 'thehindu.com'
    ]
    
    @staticmethod
    def is_news_site(url: str) -> bool:
        """Check if URL is a known news site"""
        domain = urlparse(url).netloc.lower()
        return any(news_domain in domain for news_domain in NewsPatterns.NEWS_DOMAINS)
    
    @staticmethod
    def is_aggregator(url: str) -> bool:
        """Check if URL is a news aggregator (homepage with multiple articles)"""
        aggregator_patterns = [
            'news.google.com',
            '/home',
            '/latest',
            '/trending',
            '/top-stories',
            '/headlines'
        ]
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in aggregator_patterns)

class GeminiNewsAnalyzer:
    """
    Enhanced Gemini URL analyzer with intelligent news site handling.
    
    Features:
    - Detects news aggregator pages
    - Extracts article links from main pages
    - Summarizes individual articles or entire collections
    - Filters by category, keyword, or date
    - Handles both single articles and bulk analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Set it in environment variables or pass as parameter."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "models/gemini-2.5-flash"
        
        logger.info(f"✓ Initialized GeminiNewsAnalyzer with model: {self.model_name}")
    
    def extract_urls(self, query: str) -> List[str]:
        """Extract URLs from query string"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        urls = url_pattern.findall(query)
        return list(set(urls))
    
    async def _call_gemini(self, prompt: str) -> Optional[str]:
        """Helper method to call Gemini API"""
        try:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=GenerateContentConfig(
                        tools=[{'google_search': {}}],
                        response_modalities=["TEXT"]
                    )
                )
            )
            
            if response.candidates:
                content_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        content_parts.append(part.text)
                return '\n'.join(content_parts) if content_parts else None
            return None
        
        except Exception as e:
            logger.error(f"[GEMINI] API call failed: {e}")
            return None
    
    async def extract_article_links(
        self, 
        news_url: str,
        category_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Extract article links from a news aggregator page.
        
        Args:
            news_url: Main news site URL
            category_filter: Optional category (e.g., "technology", "sports")
            limit: Maximum number of articles to extract
        
        Returns:
            List of article URLs
        """
        try:
            logger.info(f"[GEMINI] Extracting article links from: {news_url}")
            
            # Build extraction prompt
            prompt = f"""Analyze this news website and extract article links: {news_url}

Instructions:
1. List all article URLs you can find on this page
2. Return ONLY the URLs, one per line
3. Do not include navigation links, ads, or non-article pages
"""
            
            if category_filter:
                prompt += f"4. Focus on articles related to: {category_filter}\n"
            
            if limit:
                prompt += f"5. Return maximum {limit} most prominent/recent articles\n"
            
            prompt += "\nFormat: Return only URLs, one per line, nothing else."
            
            content = await self._call_gemini(prompt)
            
            if not content:
                logger.warning("[GEMINI] No content returned for article extraction")
                return []
            
            # Extract URLs from response
            urls = self.extract_urls(content)
            
            # Filter to only include URLs from the same domain (avoid external links)
            base_domain = urlparse(news_url).netloc
            filtered_urls = [
                url for url in urls 
                if urlparse(url).netloc == base_domain or 'news.google.com' in base_domain
            ]
            
            logger.info(f"[GEMINI] ✓ Extracted {len(filtered_urls)} article links")
            return filtered_urls[:limit] if limit else filtered_urls
        
        except Exception as e:
            logger.error(f"[GEMINI] Error extracting article links: {e}")
            return []
    
    async def analyze_news_site(
        self,
        url: str,
        mode: Literal["all", "top", "category", "single"] = "top",
        category: Optional[str] = None,
        num_articles: int = 5,
        custom_instruction: Optional[str] = None
    ) -> GeminiURLResponse:
        """
        Intelligently analyze a news website.
        
        Args:
            url: News website URL
            mode: Analysis mode:
                - "all": Summarize all articles found
                - "top": Summarize top N articles (default 5)
                - "category": Filter and summarize by category
                - "single": Treat as single article
            category: Category filter (for "category" mode)
            num_articles: Number of articles to analyze (for "top" mode)
            custom_instruction: Custom analysis instruction
        
        Returns:
            GeminiURLResponse with comprehensive analysis
        """
        try:
            is_news = NewsPatterns.is_news_site(url)
            is_aggregator = NewsPatterns.is_aggregator(url)
            
            logger.info(f"[GEMINI] Analyzing URL - News site: {is_news}, Aggregator: {is_aggregator}")
            
            # Single article mode
            if mode == "single" or (is_news and not is_aggregator):
                return await self.analyze_single_article(url, custom_instruction)
            
            # Multi-article modes
            if is_aggregator or mode in ["all", "top", "category"]:
                return await self.analyze_multi_articles(
                    url, mode, category, num_articles, custom_instruction
                )
            
            # Fallback to single article
            return await self.analyze_single_article(url, custom_instruction)
        
        except Exception as e:
            logger.error(f"[GEMINI] Error in analyze_news_site: {e}", exc_info=True)
            return GeminiURLResponse(
                success=False,
                error=str(e)
            )
    
    async def analyze_single_article(
        self,
        url: str,
        instruction: Optional[str] = None
    ) -> GeminiURLResponse:
        """Analyze a single article"""
        try:
            logger.info(f"[GEMINI] Analyzing single article: {url}")
            
            if not instruction:
                instruction = """Provide a comprehensive summary of this article including:
1. Main headline and topic
2. Key points and findings
3. Important quotes or data
4. Conclusion or implications"""
            
            prompt = f"{instruction}\n\nURL: {url}"
            content = await self._call_gemini(prompt)
            
            if not content:
                return GeminiURLResponse(
                    success=False,
                    error="No content generated by Gemini"
                )
            
            article = ArticleInfo(url=url, summary=content)
            
            return GeminiURLResponse(
                success=True,
                content=content,
                urls_analyzed=[url],
                articles=[article],
                is_news_site=True,
                total_articles_found=1
            )
        
        except Exception as e:
            logger.error(f"[GEMINI] Error analyzing single article: {e}")
            return GeminiURLResponse(
                success=False,
                error=str(e)
            )
    
    async def analyze_multi_articles(
        self,
        base_url: str,
        mode: str,
        category: Optional[str],
        num_articles: int,
        custom_instruction: Optional[str]
    ) -> GeminiURLResponse:
        """Analyze multiple articles from a news site"""
        try:
            logger.info(f"[GEMINI] Multi-article analysis - Mode: {mode}, Limit: {num_articles}")
            
            # Step 1: Extract article links
            category_filter = category if mode == "category" else None
            limit = num_articles if mode == "top" else None
            
            article_urls = await self.extract_article_links(
                base_url, 
                category_filter=category_filter,
                limit=limit
            )
            
            if not article_urls:
                # Fallback: Analyze the main page directly
                logger.info("[GEMINI] No article links extracted, analyzing main page")
                return await self._analyze_aggregator_page(base_url, custom_instruction)
            
            logger.info(f"[GEMINI] Found {len(article_urls)} articles to analyze")
            
            # Step 2: Build comprehensive prompt for all articles
            if custom_instruction:
                instruction = custom_instruction
            else:
                instruction = f"""Analyze the following {len(article_urls)} news articles and provide:

1. **Overview**: General theme/topics covered
2. **Individual Summaries**: Brief summary of each article (2-3 sentences)
3. **Key Insights**: Main takeaways across all articles
4. **Trends**: Any patterns or connections between articles"""
            
            if category:
                instruction += f"\n5. **Category Focus**: Emphasis on {category}-related content"
            
            # Add all article URLs to prompt
            urls_text = '\n'.join([f"Article {i+1}: {url}" for i, url in enumerate(article_urls)])
            prompt = f"{instruction}\n\n{urls_text}"
            
            # Step 3: Get comprehensive analysis
            content = await self._call_gemini(prompt)
            
            if not content:
                return GeminiURLResponse(
                    success=False,
                    error="Failed to generate multi-article analysis"
                )
            
            # Step 4: Build response
            articles = [ArticleInfo(url=url) for url in article_urls]
            
            return GeminiURLResponse(
                success=True,
                content=content,
                urls_analyzed=article_urls,
                articles=articles,
                is_news_site=True,
                total_articles_found=len(article_urls)
            )
        
        except Exception as e:
            logger.error(f"[GEMINI] Error in multi-article analysis: {e}", exc_info=True)
            return GeminiURLResponse(
                success=False,
                error=str(e)
            )
    
    async def _analyze_aggregator_page(
        self,
        url: str,
        instruction: Optional[str]
    ) -> GeminiURLResponse:
        """Fallback: Analyze aggregator page directly without extracting links"""
        try:
            if not instruction:
                instruction = """Analyze this news aggregator page and provide:
1. Main news categories covered
2. Top trending stories (titles and brief summaries)
3. Key themes across all content
4. Notable headlines or breaking news"""
            
            prompt = f"{instruction}\n\nURL: {url}"
            content = await self._call_gemini(prompt)
            
            if not content:
                return GeminiURLResponse(
                    success=False,
                    error="Failed to analyze aggregator page"
                )
            
            return GeminiURLResponse(
                success=True,
                content=content,
                urls_analyzed=[url],
                is_news_site=True,
                total_articles_found=0
            )
        
        except Exception as e:
            logger.error(f"[GEMINI] Error analyzing aggregator page: {e}")
            return GeminiURLResponse(
                success=False,
                error=str(e)
            )
    
    async def analyze_query_with_urls(
        self,
        query: str,
        auto_detect_mode: bool = True
    ) -> GeminiURLResponse:
        """
        Automatically detect URLs and handle them intelligently.
        
        Args:
            query: User query with URLs
            auto_detect_mode: Automatically determine analysis mode
        
        Returns:
            GeminiURLResponse with analysis
        """
        try:
            urls = self.extract_urls(query)
            
            if not urls:
                logger.info("[GEMINI] No URLs found in query")
                return GeminiURLResponse(
                    success=True,
                    content=query,
                    urls_analyzed=[]
                )
            
            logger.info(f"[GEMINI] Found {len(urls)} URL(s): {urls}")
            
            # Detect intent from query
            query_lower = query.lower()
            
            # Check for multi-article intent
            multi_keywords = ['all articles', 'all news', 'all topics', 'everything', 'all stories']
            top_keywords = ['top', 'latest', 'recent', 'trending', 'main']
            category_keywords = ['technology', 'sports', 'politics', 'business', 'entertainment', 
                               'science', 'health', 'world', 'local']
            
            # Extract number if specified (e.g., "top 10 articles")
            num_match = re.search(r'\b(\d+)\b', query_lower)
            num_articles = int(num_match.group(1)) if num_match else 5
            
            # Determine mode
            mode = "single"
            category = None
            
            if any(kw in query_lower for kw in multi_keywords):
                mode = "all"
            elif any(kw in query_lower for kw in top_keywords):
                mode = "top"
            else:
                # Check for category
                for cat in category_keywords:
                    if cat in query_lower:
                        mode = "category"
                        category = cat
                        break
            
            # Use the first URL
            return await self.analyze_news_site(
                url=urls[0],
                mode=mode,
                category=category,
                num_articles=num_articles,
                custom_instruction=query
            )
        
        except Exception as e:
            logger.error(f"[GEMINI] Error in analyze_query_with_urls: {e}")
            return GeminiURLResponse(
                success=False,
                error=str(e)
            )


# ============================================================================
# EXAMPLE USAGE & TESTS
# ============================================================================

async def example_usage():
    """Demonstrate different usage patterns"""
    analyzer = GeminiNewsAnalyzer()
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Analyze top 5 articles from Google News")
    print("="*70)
    
    result = await analyzer.analyze_news_site(
        url="https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN%3Aen",
        mode="top",
        num_articles=5
    )
    
    if result.success:
        print(f"✓ Found {result.total_articles_found} articles")
        print(f"\n{result.content}\n")
    else:
        print(f"✗ Error: {result.error}")
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Technology news from category filter")
    print("="*70)
    
    result = await analyzer.analyze_news_site(
        url="https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN%3Aen",
        mode="category",
        category="technology",
        num_articles=3
    )
    
    if result.success:
        print(f"✓ Technology articles analyzed")
        print(f"\n{result.content}\n")
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Natural language query")
    print("="*70)
    
    result = await analyzer.analyze_query_with_urls(
        "Summarize the top 3 sports news from https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN%3Aen"
    )
    
    if result.success:
        print(f"✓ Query processed")
        print(f"\n{result.content}\n")


class GeminiURLAnalyzer:
    """
    Compatibility wrapper for GeminiNewsAnalyzer to support legacy GeminiURLAnalyzer interface.
    Required by rag_routes.py.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.analyzer = GeminiNewsAnalyzer(api_key)
        
    async def analyze_url(self, url: str, instruction: str = "Summarize the main content") -> GeminiURLResponse:
        """Analyze a single URL"""
        return await self.analyzer.analyze_single_article(url, instruction)
        
    async def compare_urls(self, urls: List[str], comparison_instruction: Optional[str] = None) -> GeminiURLResponse:
        """Compare multiple URLs"""
        if len(urls) < 2:
             return GeminiURLResponse(success=False, error="Need at least 2 URLs for comparison")
             
        instruction = comparison_instruction or "Compare and contrast the content from these URLs."
        urls_text = "\\n".join([f"URL {i+1}: {url}" for i, url in enumerate(urls)])
        prompt = f"{instruction}\\n\\n{urls_text}"
        
        content = await self.analyzer._call_gemini(prompt)
        
        if content:
            return GeminiURLResponse(
                success=True,
                content=content,
                urls_analyzed=urls,
                is_news_site=False
            )
        return GeminiURLResponse(success=False, error="Comparison generation failed")

    async def analyze_query_with_urls(self, query: str) -> GeminiURLResponse:
        """Analyze query with URLs"""
        return await self.analyzer.analyze_query_with_urls(query)


if __name__ == "__main__":
    print("Running Enhanced Gemini News Analyzer Examples...")
    asyncio.run(example_usage())
# """
# Fixed Gemini URL Connector
# Based on verified working Colab example with gemini-2.5-flash + google_search tool
# Replace your existing src/gemini_url_connector.py with this file
# """

# import os
# import logging
# import asyncio
# from typing import Dict, Any, Optional, List
# from dataclasses import dataclass
# import re

# try:
#     from google import genai
#     from google.genai.types import GenerateContentConfig
#     GEMINI_AVAILABLE = True
# except ImportError:
#     GEMINI_AVAILABLE = False
#     logging.warning("Google Generative AI library not installed. Install with: pip install google-genai")

# logger = logging.getLogger(__name__)

# @dataclass
# class GeminiURLResponse:
#     """Response from Gemini URL analysis"""
#     success: bool
#     content: Optional[str] = None
#     urls_analyzed: List[str] = None
#     error: Optional[str] = None
#     grounding_metadata: Optional[Dict] = None

# class GeminiURLAnalyzer:
#     """
#     Uses Google's Gemini API with google_search grounding to analyze URLs.
    
#     Based on verified working implementation:
#     - Model: gemini-2.5-flash (newest model)
#     - Tool: google_search (correct tool name)
#     - Response: TEXT modality
    
#     This is much more powerful than web scraping as Gemini can:
#     - Access dynamic content
#     - Understand context and structure
#     - Compare multiple pages intelligently
#     - Extract specific information accurately
#     """
    
#     def __init__(self, api_key: Optional[str] = None):
#         if not GEMINI_AVAILABLE:
#             raise ImportError(
#                 "google-genai package not installed. "
#                 "Install with: pip install google-genai"
#             )
        
#         self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
#         if not self.api_key:
#             raise ValueError(
#                 "GOOGLE_API_KEY not found. "
#                 "Set it in environment variables or pass as parameter."
#             )
        
#         # Initialize Gemini client
#         self.client = genai.Client(api_key=self.api_key)
        
#         # Use gemini-2.5-flash (newest model with google_search support)
#         self.model_name = "models/gemini-2.5-flash"
        
#         logger.info(f"✓ Initialized GeminiURLAnalyzer with model: {self.model_name}")
    
#     def extract_urls(self, query: str) -> List[str]:
#         """Extract URLs from query string using regex"""
#         url_pattern = re.compile(
#             r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#         )
#         urls = url_pattern.findall(query)
#         return list(set(urls))  # Remove duplicates
    
#     async def analyze_url(
#         self, 
#         url: str, 
#         instruction: str = "Summarize the main content and key points from this page"
#     ) -> GeminiURLResponse:
#         """
#         Analyze a single URL using Gemini with google_search grounding.
        
#         Args:
#             url: The URL to analyze
#             instruction: What you want Gemini to do with the URL content
        
#         Returns:
#             GeminiURLResponse with analysis results
#         """
#         try:
#             logger.info(f"[GEMINI] Analyzing URL: {url}")
            
#             # Construct the prompt (matches working example format)
#             prompt = f"{instruction}\n\nURL: {url}"
            
#             # Make the request with google_search grounding (VERIFIED WORKING)
#             response = await asyncio.to_thread(
#                 lambda: self.client.models.generate_content(
#                     model=self.model_name,
#                     contents=prompt,
#                     config=GenerateContentConfig(
#                         tools=[{'google_search': {}}],  # Correct tool name
#                         response_modalities=["TEXT"]
#                     )
#                 )
#             )
            
#             # Extract the response text (matches working example)
#             if response.candidates:
#                 content_parts = []
#                 for part in response.candidates[0].content.parts:
#                     if hasattr(part, 'text') and part.text:
#                         content_parts.append(part.text)
                
#                 if not content_parts:
#                     logger.warning(f"[GEMINI] Response had candidates but no text parts for {url}")
#                     return GeminiURLResponse(
#                         success=False,
#                         error="No text content in Gemini response"
#                     )
                
#                 content = '\n'.join(content_parts)
                
#                 # Extract grounding metadata if available
#                 grounding_metadata = None
#                 if hasattr(response, 'grounding_metadata'):
#                     try:
#                         grounding_metadata = {
#                             'search_queries': getattr(response.grounding_metadata, 'search_entry_point', None),
#                             'grounding_chunks': getattr(response.grounding_metadata, 'grounding_chunks', None)
#                         }
#                     except Exception as e:
#                         logger.debug(f"Could not extract grounding metadata: {e}")
                
#                 logger.info(f"[GEMINI] ✓ Successfully analyzed {url}, response length: {len(content)} chars")
                
#                 return GeminiURLResponse(
#                     success=True,
#                     content=content,
#                     urls_analyzed=[url],
#                     grounding_metadata=grounding_metadata
#                 )
#             else:
#                 logger.warning(f"[GEMINI] No candidates in response for {url}")
#                 return GeminiURLResponse(
#                     success=False,
#                     error="No content generated by Gemini"
#                 )
        
#         except Exception as e:
#             logger.error(f"[GEMINI] Error analyzing {url}: {e}", exc_info=True)
#             return GeminiURLResponse(
#                 success=False,
#                 error=f"Gemini API error: {str(e)}"
#             )
    
#     async def compare_urls(
#         self, 
#         urls: List[str], 
#         comparison_instruction: Optional[str] = None
#     ) -> GeminiURLResponse:
#         """
#         Compare multiple URLs using Gemini.
        
#         Args:
#             urls: List of URLs to compare (2-5 recommended)
#             comparison_instruction: Specific comparison task
        
#         Returns:
#             GeminiURLResponse with comparison results
#         """
#         try:
#             if len(urls) < 2:
#                 return GeminiURLResponse(
#                     success=False,
#                     error="Need at least 2 URLs to compare"
#                 )
            
#             if len(urls) > 5:
#                 logger.warning(f"[GEMINI] Comparing {len(urls)} URLs may be slow. Consider limiting to 5.")
            
#             logger.info(f"[GEMINI] Comparing {len(urls)} URLs")
            
#             # Default comparison instruction
#             if not comparison_instruction:
#                 comparison_instruction = (
#                     f"Compare and contrast the content from these {len(urls)} URLs. "
#                     "Identify key similarities, differences, and provide a comprehensive analysis."
#                 )
            
#             # Construct prompt with all URLs (matches working format)
#             urls_text = '\n'.join([f"URL {i+1}: {url}" for i, url in enumerate(urls)])
#             prompt = f"{comparison_instruction}\n\n{urls_text}"
            
#             # Make the request
#             response = await asyncio.to_thread(
#                 lambda: self.client.models.generate_content(
#                     model=self.model_name,
#                     contents=prompt,
#                     config=GenerateContentConfig(
#                         tools=[{'google_search': {}}],
#                         response_modalities=["TEXT"]
#                     )
#                 )
#             )
            
#             # Extract response
#             if response.candidates:
#                 content_parts = []
#                 for part in response.candidates[0].content.parts:
#                     if hasattr(part, 'text') and part.text:
#                         content_parts.append(part.text)
                
#                 if not content_parts:
#                     return GeminiURLResponse(
#                         success=False,
#                         error="No text content in comparison response"
#                     )
                
#                 content = '\n'.join(content_parts)
                
#                 logger.info(f"[GEMINI] ✓ Successfully compared {len(urls)} URLs")
                
#                 return GeminiURLResponse(
#                     success=True,
#                     content=content,
#                     urls_analyzed=urls
#                 )
#             else:
#                 return GeminiURLResponse(
#                     success=False,
#                     error="No content generated for comparison"
#                 )
        
#         except Exception as e:
#             logger.error(f"[GEMINI] Error comparing URLs: {e}", exc_info=True)
#             return GeminiURLResponse(
#                 success=False,
#                 error=f"Gemini comparison error: {str(e)}"
#             )
    
#     async def analyze_query_with_urls(
#         self, 
#         query: str, 
#         extract_and_analyze: bool = True
#     ) -> GeminiURLResponse:
#         """
#         Automatically detect URLs in query and analyze them using Gemini.
        
#         Args:
#             query: User query potentially containing URLs
#             extract_and_analyze: If True, extract URLs and enhance the query
        
#         Returns:
#             GeminiURLResponse with analysis
#         """
#         try:
#             urls = self.extract_urls(query)
            
#             if not urls:
#                 # No URLs found, just process the query normally
#                 logger.info("[GEMINI] No URLs found in query")
#                 return GeminiURLResponse(
#                     success=True,
#                     content=query,
#                     urls_analyzed=[]
#                 )
            
#             logger.info(f"[GEMINI] Found {len(urls)} URL(s) in query: {urls}")
            
#             # If multiple URLs, check if it's a comparison query
#             if len(urls) > 1:
#                 comparison_keywords = [
#                     'compare', 'difference', 'versus', 'vs', 'vs.', 
#                     'contrast', 'similar', 'better', 'which'
#                 ]
#                 is_comparison = any(kw in query.lower() for kw in comparison_keywords)
                
#                 if is_comparison:
#                     logger.info("[GEMINI] Detected comparison query")
#                     return await self.compare_urls(urls, query)
            
#             # Single URL or general analysis
#             # Use the original query as instruction (remove URL from instruction)
#             instruction = query
#             for url in urls:
#                 instruction = instruction.replace(url, '').strip()
            
#             if not instruction:
#                 instruction = "Analyze the content from this URL"
            
#             return await self.analyze_url(urls[0], instruction)
        
#         except Exception as e:
#             logger.error(f"[GEMINI] Error in analyze_query_with_urls: {e}", exc_info=True)
#             return GeminiURLResponse(
#                 success=False,
#                 error=str(e)
#             )


# # ============================================================================
# # STANDALONE TEST FUNCTION
# # ============================================================================

# async def test_gemini_analyzer():
#     """Test the Gemini URL analyzer with a real example"""
#     try:
#         # Initialize with environment API key
#         analyzer = GeminiURLAnalyzer()
        
#         # Test URL from working example
#         test_url = "https://www.allrecipes.com/recipes/17562/dinner/"
#         instruction = "Compare the ingredients and cooking times from the recipes at this URL"
        
#         print(f"\n{'='*60}")
#         print(f"Testing Gemini URL Analyzer")
#         print(f"{'='*60}")
#         print(f"URL: {test_url}")
#         print(f"Instruction: {instruction}")
#         print(f"{'='*60}\n")
        
#         # Analyze URL
#         result = await analyzer.analyze_url(test_url, instruction)
        
#         if result.success:
#             print("✓ SUCCESS\n")
#             print(result.content)
#             print(f"\n{'='*60}")
#             print(f"URLs analyzed: {result.urls_analyzed}")
#             print(f"Grounding used: {result.grounding_metadata is not None}")
#             print(f"{'='*60}\n")
#         else:
#             print(f"✗ FAILED: {result.error}\n")
        
#         return result
    
#     except Exception as e:
#         print(f"✗ TEST FAILED: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# # ============================================================================
# # RUN TEST IF EXECUTED DIRECTLY
# # ============================================================================

# if __name__ == "__main__":
#     import asyncio
    
#     print("Running Gemini URL Analyzer test...")
#     result = asyncio.run(test_gemini_analyzer())
    
#     if result and result.success:
#         print("\n✓ All tests passed!")
#     else:
#         print("\n✗ Tests failed!")