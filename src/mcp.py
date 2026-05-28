import asyncio
import json
import time
import logging
import requests
import yfinance as yf
import arxiv
import sympy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
import asyncio
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import json
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from langchain.chains import LLMMathChain
import re
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, urljoin, quote_plus
from dataclasses import dataclass
import random
import requests
import cloudscraper
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ddgs import DDGS
import sympy
from src.snapshot_operations import download_snapshot, poll_snapshot_status
# Configure logging
logger = logging.getLogger(__name__)

# --- Standardized Data Models ---

class ConnectionStatus(Enum):
    """Enumeration for the health status of a connector."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ConnectorResponse:
    """A standardized response object from any connector."""
    success: bool
    source: str
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# --- Abstract Base Class for Connectors ---
class BaseConnector(ABC):
    """
    An abstract base class that all tool connectors must inherit from.
    It provides a standard interface with built-in metrics and health tracking.
    """
    def __init__(self, name: str, timeout: float = 15.0):
        self.name = name
        self.timeout = timeout
        self._status = ConnectionStatus.UNKNOWN
        self.total_requests = 0
        self.failed_requests = 0

    @abstractmethod
    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        """The core logic for the connector must be implemented by subclasses."""
        pass

    async def __call__(self, params: Dict[str, Any]) -> ConnectorResponse:
        """Executes the connector with timeout, error handling, and metrics."""
        self.total_requests += 1
        start_time = time.time()
        try:
            response = await asyncio.wait_for(self._execute(params), timeout=self.timeout)
            self._status = ConnectionStatus.HEALTHY
            response.metadata['execution_time_ms'] = round((time.time() - start_time) * 1000, 2)
            return response
        except asyncio.TimeoutError:
            self.failed_requests += 1
            self._status = ConnectionStatus.UNHEALTHY
            logger.error(f"Connector '{self.name}' timed out.")
            return ConnectorResponse(success=False, source=self.name, error="Request timed out.")
        except Exception as e:
            self.failed_requests += 1
            self._status = ConnectionStatus.DEGRADED
            logger.error(f"Error in connector '{self.name}': {e}")
            return ConnectorResponse(success=False, source=self.name, error=str(e))

    def get_health(self) -> Dict[str, Any]:
        """Returns the current health and metrics of the connector."""
        return {
            "name": self.name,
            "status": self._status.value,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
        }

# --- CORRECTED: BrightDataRedditSearchConnector ---
class BrightDataRedditSearchConnector(BaseConnector):
    """
    Connector for Reddit keyword search via Bright Data.
    Now properly implements snapshot polling and data retrieval.
    """
    def __init__(self):
        super().__init__("reddit_search_brightdata", timeout=120.0)  # Increased timeout for polling
        self.trigger_url = "https://api.brightdata.com/datasets/v3/trigger"
        self.dataset_id = "gd_lvz8ah06191smkebj4"

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        keyword = params.get("keyword")
        if not keyword:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="'keyword' parameter is required."
            )

        logger.info(f"Triggering Reddit search for keyword: '{keyword}'")

        trigger_params = {
            "dataset_id": self.dataset_id,
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "keyword"
        }

        data = [{
            "keyword": keyword,
            "date": params.get("date", "All time"),
            "sort_by": params.get("sort_by", "Hot"),
            "num_of_posts": params.get("num_of_posts", 75),
        }]

        raw_data = await self._trigger_and_download_snapshot(
            self.trigger_url,
            trigger_params,
            data
        )

        if raw_data is None:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="Failed to trigger or download Reddit search snapshot. Check API key and dataset permissions."
            )

        # Validate and parse the data
        if not isinstance(raw_data, list):
            logger.error(f"Unexpected data format from Bright Data: {type(raw_data)}")
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="Received invalid data format from Bright Data API"
            )

        if len(raw_data) == 0:
            logger.warning(f"No Reddit posts found for keyword: '{keyword}'")
            return ConnectorResponse(
                success=True,
                source=self.name,
                data={"parsed_posts": [], "total_found": 0},
                metadata={"keyword": keyword, "message": "No posts found"}
            )

        # Parse and validate post data
        parsed_posts = []
        for post in raw_data:
            if not isinstance(post, dict):
                logger.warning(f"Skipping invalid post data: {post}")
                continue

            title = post.get("title", "")
            url = post.get("url", "")

            if not title and not url:
                logger.warning(f"Skipping post with missing title and URL")
                continue

            parsed_post = {
                "title": title or "[No title]",
                "url": url or "",
                "subreddit": post.get("subreddit", "unknown"),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "author": post.get("author", "[deleted]"),
                "created_utc": post.get("created_utc", ""),
                "selftext": post.get("selftext", "")[:500] if post.get("selftext") else ""
            }
            parsed_posts.append(parsed_post)

        logger.info(f"Successfully parsed {len(parsed_posts)} Reddit posts for '{keyword}'")

        return ConnectorResponse(
            success=True,
            source=self.name,
            data={
                "parsed_posts": parsed_posts,
                "total_found": len(parsed_posts),
                "keyword": keyword
            },
            metadata={
                "raw_count": len(raw_data),
                "parsed_count": len(parsed_posts)
            }
        )

    async def _trigger_and_download_snapshot(
        self,
        url: str,
        params: Dict[str, Any],
        data: list
    ) -> Optional[list]:
        """
        Triggers a Bright Data snapshot and waits for it to complete.
        Returns the downloaded data or None on failure.
        """
        api_key = os.getenv("BRIGHTDATA_API_KEY")
        if not api_key:
            logger.error("BRIGHTDATA_API_KEY not found in environment variables")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        loop = asyncio.get_event_loop()

        try:
            # Step 1: Trigger the snapshot
            logger.info(f"Triggering Bright Data snapshot with params: {params}")
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, params=params, json=data, timeout=30)
            )
            response.raise_for_status()
            trigger_result = response.json()

            snapshot_id = trigger_result.get("snapshot_id")
            if not snapshot_id:
                logger.error(f"No snapshot_id in trigger response: {trigger_result}")
                return None

            logger.info(f"Successfully triggered snapshot: {snapshot_id}")

            # Step 2: Poll for snapshot completion
            logger.info(f"Polling snapshot status for: {snapshot_id}")
            is_ready = await loop.run_in_executor(
                None,
                lambda: poll_snapshot_status(snapshot_id, max_attempts=60, poll_interval=5)
            )

            if not is_ready:
                logger.error(f"Snapshot {snapshot_id} failed or timed out during polling")
                return None

            # Step 3: Download the snapshot data
            logger.info(f"Downloading snapshot data for: {snapshot_id}")
            snapshot_data = await loop.run_in_executor(
                None,
                lambda: download_snapshot(snapshot_id)
            )

            if snapshot_data is None:
                logger.error(f"Failed to download snapshot data for: {snapshot_id}")
                return None

            logger.info(f"Successfully downloaded snapshot {snapshot_id} with {len(snapshot_data) if isinstance(snapshot_data, list) else '?'} records")
            return snapshot_data

        except requests.HTTPError as e:
            logger.error(f"HTTP error triggering Bright Data snapshot: {e.response.status_code}")
            try:
                error_details = e.response.json()
                logger.error(f"Error details: {error_details}")
            except:
                logger.error(f"Response text: {e.response.text}")
            return None

        except requests.RequestException as e:
            logger.error(f"Network error during snapshot operation: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in snapshot operation: {e}", exc_info=True)
            return None


# --- CORRECTED: BrightDataRedditPostRetrievalConnector ---
class BrightDataRedditPostRetrievalConnector(BaseConnector):
    """
    Connector for retrieving Reddit post comments via Bright Data.
    Now properly implements snapshot polling and data retrieval.
    """
    def __init__(self):
        super().__init__("reddit_post_retrieval_brightdata", timeout=120.0)  # Increased timeout
        self.trigger_url = "https://api.brightdata.com/datasets/v3/trigger"
        self.dataset_id = "gd_lvzdpsdlw09j6t702"

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        urls = params.get("urls")
        if not urls:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="'urls' parameter is required (list of Reddit post URLs)."
            )

        if not isinstance(urls, list):
            urls = [urls]

        logger.info(f"Triggering comment retrieval for {len(urls)} Reddit post(s)")

        trigger_params = {
            "dataset_id": self.dataset_id,
            "include_errors": "true"
        }

        data = [{
            "url": url,
            "days_back": params.get("days_back", 10),
            "load_all_replies": params.get("load_all_replies", False),
            "comment_limit": params.get("comment_limit", "")
        } for url in urls]

        raw_data = await self._trigger_and_download_snapshot(
            self.trigger_url,
            trigger_params,
            data
        )

        if raw_data is None:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="Failed to trigger or download Reddit comments snapshot. Check API key and dataset permissions."
            )

        # Validate and parse the data
        if not isinstance(raw_data, list):
            logger.error(f"Unexpected data format from Bright Data: {type(raw_data)}")
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="Received invalid data format from Bright Data API"
            )

        if len(raw_data) == 0:
            logger.warning(f"No comments found for provided URLs")
            return ConnectorResponse(
                success=True,
                source=self.name,
                data={"comments": [], "total_retrieved": 0},
                metadata={"urls": urls, "message": "No comments found"}
            )

        # Parse and validate comment data
        parsed_comments = []
        for comment in raw_data:
            if not isinstance(comment, dict):
                logger.warning(f"Skipping invalid comment data: {comment}")
                continue

            comment_id = comment.get("comment_id", "")
            comment_text = comment.get("comment", "")

            if not comment_id and not comment_text:
                logger.warning(f"Skipping comment with missing ID and text")
                continue

            parsed_comment = {
                "comment_id": comment_id or "unknown",
                "content": comment_text or "[No content]",
                "date": comment.get("date_posted", ""),
                "author": comment.get("author", "[deleted]"),
                "score": comment.get("score", 0),
                "parent_id": comment.get("parent_id", ""),
                "post_url": comment.get("post_url", "")
            }
            parsed_comments.append(parsed_comment)

        logger.info(f"Successfully parsed {len(parsed_comments)} Reddit comments")

        return ConnectorResponse(
            success=True,
            source=self.name,
            data={
                "comments": parsed_comments,
                "total_retrieved": len(parsed_comments),
                "urls": urls
            },
            metadata={
                "raw_count": len(raw_data),
                "parsed_count": len(parsed_comments)
            }
        )

    async def _trigger_and_download_snapshot(
        self,
        url: str,
        params: Dict[str, Any],
        data: list
    ) -> Optional[list]:
        """
        Triggers a Bright Data snapshot and waits for it to complete.
        Returns the downloaded data or None on failure.
        """
        api_key = os.getenv("BRIGHTDATA_API_KEY")
        if not api_key:
            logger.error("BRIGHTDATA_API_KEY not found in environment variables")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        loop = asyncio.get_event_loop()

        try:
            # Step 1: Trigger the snapshot
            logger.info(f"Triggering Bright Data comments snapshot with params: {params}")
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, params=params, json=data, timeout=30)
            )
            response.raise_for_status()
            trigger_result = response.json()

            snapshot_id = trigger_result.get("snapshot_id")
            if not snapshot_id:
                logger.error(f"No snapshot_id in trigger response: {trigger_result}")
                return None

            logger.info(f"Successfully triggered comments snapshot: {snapshot_id}")

            # Step 2: Poll for snapshot completion
            logger.info(f"Polling comments snapshot status for: {snapshot_id}")
            is_ready = await loop.run_in_executor(
                None,
                lambda: poll_snapshot_status(snapshot_id, max_attempts=60, poll_interval=5)
            )

            if not is_ready:
                logger.error(f"Comments snapshot {snapshot_id} failed or timed out during polling")
                return None

            # Step 3: Download the snapshot data
            logger.info(f"Downloading comments snapshot data for: {snapshot_id}")
            snapshot_data = await loop.run_in_executor(
                None,
                lambda: download_snapshot(snapshot_id)
            )

            if snapshot_data is None:
                logger.error(f"Failed to download comments snapshot data for: {snapshot_id}")
                return None

            logger.info(f"Successfully downloaded comments snapshot {snapshot_id} with {len(snapshot_data) if isinstance(snapshot_data, list) else '?'} records")
            return snapshot_data

        except requests.HTTPError as e:
            logger.error(f"HTTP error triggering Bright Data comments snapshot: {e.response.status_code}")
            try:
                error_details = e.response.json()
                logger.error(f"Error details: {error_details}")
            except:
                logger.error(f"Response text: {e.response.text}")
            return None

        except requests.RequestException as e:
            logger.error(f"Network error during comments snapshot operation: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in comments snapshot operation: {e}", exc_info=True)
            return None
# --- Concrete Connector Implementations ---
class TavilySearchConnector(BaseConnector):
    """Connector for the Tavily web search API."""
    def __init__(self, api_key: str):
        super().__init__("tavily_web_search")
        self.tavily_search = TavilySearch(tavily_api_key=api_key, max_results=5)

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(success=False, source=self.name, error="'query' parameter is required.")
        
        result = await asyncio.to_thread(self.tavily_search.run, query)
        return ConnectorResponse(success=True, source=self.name, data=result)

class DuckDuckGoSearchConnector(BaseConnector):
    """Connector for the DuckDuckGo web search API."""
    def __init__(self):
        super().__init__("duckduckgo_web_search")
        self.ddg_search = DuckDuckGoSearchRun()

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(success=False, source=self.name, error="'query' parameter is required.")
        
        result = await asyncio.to_thread(self.ddg_search.run, query)
        return ConnectorResponse(success=True, source=self.name, data=result)

class WikipediaConnector(BaseConnector):
    """Connector for the Wikipedia API."""
    def __init__(self):
        super().__init__("wikipedia_search")
        self.wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(success=False, source=self.name, error="'query' parameter is required.")

        result = await asyncio.to_thread(self.wiki_search.run, query)
        return ConnectorResponse(success=True, source=self.name, data=result)

class SymPySolverConnector(BaseConnector):
    """Connector for solving algebraic equations using SymPy."""
    def __init__(self):
        super().__init__("sympy_solver")

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        equation_str = params.get("equation")
        if not equation_str:
            return ConnectorResponse(success=False, source=self.name, error="'equation' parameter is required.")

        try:
            if '=' in equation_str:
                lhs_str, rhs_str = equation_str.split('=', 1)
                lhs = sympy.sympify(lhs_str.strip())
                rhs = sympy.sympify(rhs_str.strip())
                equation = sympy.Eq(lhs, rhs)
            else:
                return ConnectorResponse(success=False, source=self.name, error="Invalid equation format. Must contain '='.")

            variables = equation.free_symbols
            if len(variables) != 1:
                return ConnectorResponse(success=False, source=self.name, error=f"Equation must contain exactly one variable, but found {len(variables)}.")
            
            variable = list(variables)[0]

            solution = sympy.solve(equation, variable)

            if not solution:
                return ConnectorResponse(success=True, source=self.name, data=f"No solution found for the equation: {equation_str}")
            
            result = f"The solution for {variable} in the equation '{equation_str}' is {solution[0]}"
            return ConnectorResponse(success=True, source=self.name, data=result)

        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, error=f"Failed to solve equation: {e}")

class MathConnector(BaseConnector):
    """Connector for performing mathematical calculations using the reliable SymPy library."""
    def __init__(self, llm: Any):
        super().__init__("advanced_calculator")

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        expression = params.get("expression")
        if not expression or not isinstance(expression, str):
            return ConnectorResponse(success=False, source=self.name, error="'expression' parameter (string) is required.")

        try:
            processed_expression = expression.replace('^', '**')
            sympy_expr = sympy.sympify(processed_expression)
            result = sympy_expr.evalf()
            
            if isinstance(result, sympy.Float):
                result = round(float(result), 4)

            return ConnectorResponse(success=True, source=self.name, data=f"Calculation result: {result}")

        except (sympy.SympifyError, TypeError) as e:
            return ConnectorResponse(success=False, source=self.name, error=f"Invalid mathematical expression: {e}")
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, error=f"A critical error occurred: {str(e)}")

class YahooFinanceConnector(BaseConnector):
    """Connector for fetching financial data using the yfinance library."""
    def __init__(self):
        super().__init__("yahoo_finance")

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        ticker_symbol = params.get("ticker")
        if not ticker_symbol:
            return ConnectorResponse(success=False, source=self.name, error="'ticker' parameter is required.")

        ticker = yf.Ticker(ticker_symbol)
        try:
            hist = await asyncio.to_thread(lambda: ticker.history(period="1d"))
            if hist.empty:
                return ConnectorResponse(success=False, source=self.name, error=f"Invalid or delisted ticker symbol: {ticker_symbol}")

            info = await asyncio.to_thread(lambda: ticker.info)
            price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            company_name = info.get('longName', ticker_symbol)
            currency = info.get('currency', 'USD')

            if price == 'N/A':
                 return ConnectorResponse(success=False, source=self.name, error=f"Could not retrieve current price for {company_name} ({ticker_symbol}).")

            summary = f"The current stock price for {company_name} ({ticker_symbol}) is {price} {currency}."
            return ConnectorResponse(success=True, source=self.name, data=summary)
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, error=f"Failed to fetch data for {ticker_symbol}: {e}")

class WeatherConnector(BaseConnector):
    """Connector for fetching current and forecast weather from OpenWeatherMap."""
    def __init__(self, api_key: str):
        super().__init__("weather_forecast")
        self.api_key = api_key
        self.current_weather_url = "http://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "http://api.openweathermap.org/data/2.5/forecast"

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        city = params.get("city")
        if not city:
            return ConnectorResponse(success=False, source=self.name, error="'city' parameter is required.")
        
        # Optional parameter to specify which day (default to "today")
        day = params.get("day", "today").lower()
        
        if day == "today" or day == "current":
            # Get current weather
            return await self._get_current_weather(city)
        elif day == "tomorrow":
            # Get tomorrow's forecast
            return await self._get_tomorrow_forecast(city)
        else:
            return ConnectorResponse(success=False, source=self.name, 
                                   error="'day' parameter must be 'today', 'current', or 'tomorrow'")

    async def _get_current_weather(self, city: str) -> ConnectorResponse:
        """Get current weather for today."""
        request_params = {"q": city, "appid": self.api_key, "units": "metric"}
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(self.current_weather_url, params=request_params)
            )
            response.raise_for_status()
            data = response.json()
            
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            temp_min = data['main']['temp_min']
            temp_max = data['main']['temp_max']
            humidity = data['main']['humidity']
            description = data['weather'][0]['description']
            
            summary = (f"Current weather in {city.title()}: {description.title()}, "
                      f"temperature {temp:.1f}°C (feels like {feels_like:.1f}°C), "
                      f"today's range {temp_min:.1f}°C to {temp_max:.1f}°C, "
                      f"humidity {humidity}%.")
            
            return ConnectorResponse(success=True, source=self.name, data=summary)
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return ConnectorResponse(success=False, source=self.name, 
                                       error=f"City '{city}' not found.")
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get('message', e.response.text)
            return ConnectorResponse(success=False, source=self.name, 
                                   error=f"Could not find weather for {city}: {error_message}")
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, 
                                   error=f"An error occurred: {e}")

    async def _get_tomorrow_forecast(self, city: str) -> ConnectorResponse:
        """Get forecast for tomorrow."""
        request_params = {"q": city, "appid": self.api_key, "units": "metric", "cnt": 16}
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(self.forecast_url, params=request_params)
            )
            response.raise_for_status()
            data = response.json()
            
            now = datetime.now()
            tomorrow_date = (now + timedelta(days=1)).date()
            
            tomorrow_forecasts = []
            for forecast in data.get('list', []):
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                if forecast_time.date() == tomorrow_date:
                    tomorrow_forecasts.append(forecast)

            if not tomorrow_forecasts:
                return ConnectorResponse(success=True, source=self.name, 
                                       data=f"Could not find a specific forecast for tomorrow in {city.title()}.")

            min_temp = min(f['main']['temp_min'] for f in tomorrow_forecasts)
            max_temp = max(f['main']['temp_max'] for f in tomorrow_forecasts)
            conditions = [f['weather'][0]['description'] for f in tomorrow_forecasts]
            dominant_condition = max(set(conditions), key=conditions.count)
            
            summary = (f"Tomorrow's weather forecast for {city.title()}: {dominant_condition}, "
                      f"with a high of {max_temp:.1f}°C and a low of {min_temp:.1f}°C.")
            
            return ConnectorResponse(success=True, source=self.name, data=summary)
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return ConnectorResponse(success=False, source=self.name, 
                                       error=f"City '{city}' not found.")
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get('message', e.response.text)
            return ConnectorResponse(success=False, source=self.name, 
                                   error=f"Could not find weather for {city}: {error_message}")
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, 
                                   error=f"An error occurred: {e}")
        
class NewsConnector(BaseConnector):
    """
    Enhanced news connector that includes article content/descriptions.
    """
    def __init__(self, api_key: str):
        super().__init__("news_headlines", timeout=15.0)
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error="'query' parameter is required."
            )
        
        page_size = min(params.get("page_size", 10), 15)
        language = params.get("language", "en")
        sort_by = params.get("sort_by", "relevancy")  # relevancy, popularity, publishedAt
        
        request_params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": page_size,
            "sortBy": sort_by,
            "language": language
        }
        
        #
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(self.base_url, params=request_params, timeout=10)
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "ok":
                error_msg = data.get("message", "Unknown error")
                return ConnectorResponse(
                    success=False,
                    source=self.name,
                    error=f"NewsAPI error: {error_msg}"
                )
            
            articles = data.get("articles", [])
            if not articles:
                return ConnectorResponse(
                    success=True, 
                    source=self.name, 
                    data=f"No recent news articles found for '{query}'."
                )

            # Enhanced formatting with content
            formatted_articles = []
            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')
                source_name = article.get('source', {}).get('name', 'Unknown source')
                author = article.get('author', 'Unknown author')
                published_at = article.get('publishedAt', '')[:10]
                description = article.get('description', 'No description available')
                content = article.get('content', '')
                url = article.get('url', '')
                
                # Use content if available, fallback to description
                article_text = content if content else description
                if article_text and len(article_text) > 2500:
                    article_text = article_text[:2500] + "..."
                
                formatted_articles.append(
                    f"{i}. {title}\n"
                    f"   Source: {source_name} | Author: {author} | Published: {published_at}\n"
                    f"   Summary: {article_text}\n"
                    f"   URL: {url}"
                )
            
            result = (
                f"Recent news articles for '{query}':\n\n" + 
                "\n\n".join(formatted_articles)
            )
            
            return ConnectorResponse(
                success=True, 
                source=self.name, 
                data=result,
                metadata={
                    "total_results": data.get("totalResults", 0),
                    "articles_returned": len(articles)
                }
            )
            
        except requests.HTTPError as e:
            if e.response.status_code == 426:
                return ConnectorResponse(
                    success=False,
                    source=self.name,
                    error="NewsAPI requires upgrade for this endpoint."
                )
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"NewsAPI HTTP error: {e.response.status_code}"
            )
        except Exception as e:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error=f"Failed to fetch news: {str(e)}"
            )
        
class ArXivConnector(BaseConnector):
    """Connector for searching academic papers on ArXiv."""
    def __init__(self, client=None):
        super().__init__("arxiv_search", timeout=30.0)
        self.client = client if client else arxiv.Client()

    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(success=False, source=self.name, error="'query' parameter is required.")
        
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        
        
        try:
            results = await asyncio.to_thread(list, self.client.results(search))
            if not results:
                return ConnectorResponse(success=True, source=self.name, data=f"No papers found on ArXiv for '{query}'.")

            papers = [
                f"- Title: {result.title}\n  Authors: {', '.join(str(a) for a in result.authors)}\n  URL: {result.pdf_url}"
                for result in results
            ]
            summary = f"Recent ArXiv papers for '{query}':\n" + "\n\n".join(papers)
            return ConnectorResponse(success=True, source=self.name, data=summary)
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, error=f"Failed to search ArXiv: {e}")

class SemanticScholarConnector(BaseConnector):
    """Connector for searching academic papers via Semantic Scholar API (free, no API key needed)."""
    
    def __init__(self):
        super().__init__("semantic_scholar", timeout=20.0)
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(success=False, source=self.name, error="'query' parameter is required.")
        
        # Semantic Scholar API parameters
        search_params = {
            "query": query,
            "limit": 10,
            "fields": "title,authors,year,abstract,url,citationCount,publicationDate"
        }
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(self.base_url, params=search_params, timeout=15)
            )
            response.raise_for_status()
            data = response.json()
            
            papers = data.get("data", [])
            if not papers:
                return ConnectorResponse(
                    success=True, 
                    source=self.name, 
                    data=f"No academic papers found for '{query}'."
                )
            
            # Format results
            formatted_papers = []
            for i, paper in enumerate(papers, 1):
                authors = ", ".join([a.get("name", "Unknown") for a in paper.get("authors", [])[:3]])
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."
                
                title = paper.get("title", "Untitled")
                year = paper.get("year", "N/A")
                citations = paper.get("citationCount", 0)
                abstract = paper.get("abstract", "No abstract available.")[:3000]
                url = paper.get("url", "No URL")
                
                formatted_papers.append(
                    f"{i}. {title}\n"
                    f"   Authors: {authors}\n"
                    f"   Year: {year} | Citations: {citations}\n"
                    f"   Abstract: {abstract}...\n"
                    f"   URL: {url}"
                )
            
            result = f"Academic papers from Semantic Scholar for '{query}':\n\n" + "\n\n".join(formatted_papers)
            return ConnectorResponse(success=True, source=self.name, data=result)
            
        except requests.HTTPError as e:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error=f"API error: {e.response.status_code}"
            )
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, error=f"Search failed: {str(e)}")


class PubMedConnector(BaseConnector):
    """Connector for searching biomedical literature via PubMed API (free, no API key needed)."""
    
    def __init__(self):
        super().__init__("pubmed_search", timeout=20.0)
        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(success=False, source=self.name, error="'query' parameter is required.")
        
        # loop = asyncio.get_event_loop()
        
        try:
            # Step 1: Search for papers
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": 5,
                "retmode": "json",
                "sort": "relevance"
            }
            
            search_response = await asyncio.to_thread(
                lambda: requests.get(self.search_url, params=search_params, timeout=10)
            )
            search_response.raise_for_status()
            search_data = search_response.json()
            
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data=f"No biomedical papers found in PubMed for '{query}'."
                )
            
            # Step 2: Fetch paper details
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            
            fetch_response = await asyncio.to_thread(
                lambda: requests.get(self.fetch_url, params=fetch_params, timeout=10)
            )
            fetch_response.raise_for_status()
            fetch_data = fetch_response.json()
            
            papers = fetch_data.get("result", {})
            
            # Format results
            formatted_papers = []
            for i, pmid in enumerate(id_list, 1):
                paper = papers.get(pmid, {})
                title = paper.get("title", "Untitled")
                authors = paper.get("authors", [])
                author_names = ", ".join([a.get("name", "") for a in authors[:3]])
                if len(authors) > 3:
                    author_names += " et al."
                
                pub_date = paper.get("pubdate", "Unknown date")
                journal = paper.get("fulljournalname", "Unknown journal")
                
                formatted_papers.append(
                    f"{i}. {title}\n"
                    f"   Authors: {author_names}\n"
                    f"   Published: {pub_date} in {journal}\n"
                    f"   PubMed ID: {pmid}\n"
                    f"   URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                )
            
            result = f"Biomedical research papers from PubMed for '{query}':\n\n" + "\n\n".join(formatted_papers)
            return ConnectorResponse(success=True, source=self.name, data=result)
            
        except Exception as e:
            return ConnectorResponse(success=False, source=self.name, error=f"PubMed search failed: {str(e)}")


class RedditSearchConnector(BaseConnector):
    """
    Enhanced Reddit search connector with proper JSON parsing, comment extraction,
    and content filtering.
    """
    
    def __init__(self):
        super().__init__("reddit_search", timeout=20.0)
        self.base_url = "https://www.reddit.com/search.json"
        self.comment_url_template = "https://www.reddit.com{}.json"
    
    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error="'query' parameter is required."
            )
        
        include_comments = params.get("include_comments", True)
        max_comments_per_post = params.get("max_comments_per_post", 10)
        
        search_params = {
            "q": query,
            "sort": "relevance",
            "limit": 5,
            "t": "month",
            "type": "link"
        }
        
        headers = {
            "User-Agent": "ResearchBot/2.0 (Educational Research Tool)"
        }
        
        # loop = asyncio.get_event_loop()
        
        try:
            # Main search request
            response = await asyncio.to_thread(
                lambda: requests.get(
                    self.base_url, 
                    params=search_params, 
                    headers=headers, 
                    timeout=15
                )
            )
            response.raise_for_status()
            data = response.json()
            
            posts = data.get("data", {}).get("children", [])
            if not posts:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data=f"No recent Reddit discussions found for '{query}'."
                )
            
            # Process posts with proper parsing
            formatted_posts = []
            for i, post_wrapper in enumerate(posts, 1):
                post_data = post_wrapper.get("data", {})
                
                # Extract core metadata
                title = post_data.get("title", "No title")
                subreddit = post_data.get("subreddit", "unknown")
                score = post_data.get("score", 0)
                num_comments = post_data.get("num_comments", 0)
                permalink = post_data.get("permalink", "")
                url = f"https://reddit.com{permalink}"
                author = post_data.get("author", "[deleted]")
                created_utc = post_data.get("created_utc", 0)
                
                # Format timestamp
                created_date = datetime.fromtimestamp(created_utc).strftime("%Y-%m-%d")
                
                # Get post content (self text)
                selftext = post_data.get("selftext", "").strip()
                if selftext:
                    selftext = self._clean_text(selftext)[:5000]
                else:
                    selftext = "[Link post - no text content]"
                
                # Build post summary
                post_summary = (
                    f"{i}. {title}\n"
                    f"   Posted by u/{author} in r/{subreddit} on {created_date}\n"
                    f"   Score: {score} | Comments: {num_comments}\n"
                    f"   Content: {selftext}\n"
                )
                
                # Fetch top comments if requested
                if include_comments and num_comments > 0:
                    comments = await self._fetch_top_comments(
                        permalink, 
                        max_comments_per_post, 
                        headers, 
                        # loop
                    )
                    if comments:
                        post_summary += f"   Top Comments:\n{comments}\n"
                
                post_summary += f"   URL: {url}"
                formatted_posts.append(post_summary)
            
            result = f"Reddit discussions about '{query}':\n\n" + "\n\n".join(formatted_posts)
            
            return ConnectorResponse(
                success=True, 
                source=self.name, 
                data=result,
                metadata={
                    "posts_analyzed": len(posts),
                    "included_comments": include_comments
                }
            )
            
        except requests.HTTPError as e:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error=f"Reddit API error: {e.response.status_code}"
            )
        except Exception as e:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error=f"Reddit search failed: {str(e)}"
            )
    
    async def _fetch_top_comments(
        self, 
        permalink: str, 
        max_comments: int, 
        headers: Dict, 
        loop
    ) -> str:
        """Fetch and parse top comments from a post."""
        try:
            comment_url = self.comment_url_template.format(permalink)
            
            response = await asyncio.to_thread(
                lambda: requests.get(comment_url, headers=headers, timeout=10)
            )
            response.raise_for_status()
            comment_data = response.json()
            
            # Reddit returns [post_data, comments_data]
            if len(comment_data) < 2:
                return ""
            
            comments_listing = comment_data[1].get("data", {}).get("children", [])
            
            formatted_comments = []
            for comment_wrapper in comments_listing[:max_comments]:
                comment = comment_wrapper.get("data", {})
                
                # Skip non-comment items (like "more comments" links)
                if comment.get("kind") != "t1" and comment_wrapper.get("kind") != "t1":
                    continue
                
                body = comment.get("body", "")
                if not body or body in ["[deleted]", "[removed]"]:
                    continue
                
                author = comment.get("author", "[deleted]")
                score = comment.get("score", 0)
                
                cleaned_body = self._clean_text(body)[:1500]
                formatted_comments.append(
                    f"     • u/{author} ({score} pts): {cleaned_body}"
                )
            
            return "\n".join(formatted_comments) if formatted_comments else ""
            
        except Exception as e:
            logger.debug(f"Could not fetch comments: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that cause issues
        text = text.replace('\n', ' ').replace('\r', '')
        return text.strip()

class YouTubeSearchConnector(BaseConnector):
    """
    Properly scoped YouTube search connector that returns video metadata.
    For transcript extraction, use a separate YouTubeTranscriptConnector with
    the youtube-transcript-api library.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("youtube_search", timeout=15.0)
        self.api_key = api_key
        self.use_api = api_key is not None
        
        if self.use_api:
            self.base_url = "https://www.googleapis.com/youtube/v3/search"
        else:
            logger.warning(
                "YouTube API key not provided. Using fallback scraping method "
                "(less reliable, may break)."
            )
    
    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error="'query' parameter is required."
            )
        
        max_results = min(params.get("max_results", 5), 10)
        
        if self.use_api:
            return await self._search_with_api(query, max_results)
        else:
            return await self._search_with_scraping(query, max_results)
    
    async def _search_with_api(self, query: str, max_results: int) -> ConnectorResponse:
        """Use official YouTube Data API v3."""
        search_params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": self.api_key,
            "order": "relevance",
            "safeSearch": "moderate"
        }
        
        # loop = asyncio.get_event_loop()
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(self.base_url, params=search_params, timeout=10)
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data=f"No YouTube videos found for '{query}'."
                )
            
            # Format results
            formatted_videos = []
            for i, item in enumerate(items, 1):
                video_id = item.get("id", {}).get("videoId", "")
                snippet = item.get("snippet", {})
                
                title = snippet.get("title", "No title")
                channel = snippet.get("channelTitle", "Unknown channel")
                description = snippet.get("description", "No description")[:1000]
                published_at = snippet.get("publishedAt", "")[:10]
                
                formatted_videos.append(
                    f"{i}. {title}\n"
                    f"   Channel: {channel}\n"
                    f"   Published: {published_at}\n"
                    f"   Description: {description}...\n"
                    f"   URL: https://www.youtube.com/watch?v={video_id}"
                )
            
            result = f"YouTube videos about '{query}':\n\n" + "\n\n".join(formatted_videos)
            
            return ConnectorResponse(
                success=True, 
                source=self.name, 
                data=result,
                metadata={"method": "api", "results_count": len(items)}
            )
            
        except requests.HTTPError as e:
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get("error", {}).get("message", str(e))
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"YouTube API error: {error_message}"
            )
        except Exception as e:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"YouTube search failed: {str(e)}"
            )
    
    async def _search_with_scraping(self, query: str, max_results: int) -> ConnectorResponse:
        """Fallback scraping method (less reliable)."""
        search_url = "https://www.youtube.com/results"
        search_params = {"search_query": query}
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # loop = asyncio.get_event_loop()
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(
                    search_url, 
                    params=search_params, 
                    headers=headers, 
                    timeout=10
                )
            )
            response.raise_for_status()
            
            # Try to extract video data from page
            video_data = self._extract_videos_from_html(response.text, max_results)
            
            if not video_data:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data=f"No YouTube videos found for '{query}' (scraping method may be unreliable)."
                )
            
            # Format results
            formatted_videos = []
            for i, video in enumerate(video_data, 1):
                formatted_videos.append(
                    f"{i}. {video['title']}\n"
                    f"   Channel: {video['channel']}\n"
                    f"   Views: {video.get('views', 'N/A')}\n"
                    f"   URL: {video['url']}"
                )
            
            result = (
                f"YouTube videos about '{query}' (via scraping - limited info):\n\n" + 
                "\n\n".join(formatted_videos)
            )
            
            return ConnectorResponse(
                success=True, 
                source=self.name, 
                data=result,
                metadata={"method": "scraping", "results_count": len(video_data)}
            )
            
        except Exception as e:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"YouTube scraping failed: {str(e)}. Consider providing a YouTube API key."
            )
    
    def _extract_videos_from_html(self, html: str, max_results: int) -> List[Dict]:
        """Extract video information from YouTube search page HTML."""
        videos = []
        
        try:
            # Look for ytInitialData JSON
            match = re.search(r'var ytInitialData = ({.*?});', html, re.DOTALL)
            if not match:
                return videos
            
            data = json.loads(match.group(1))
            
            # Navigate JSON structure (this can break if YouTube changes their layout)
            contents = (
                data.get('contents', {})
                .get('twoColumnSearchResultsRenderer', {})
                .get('primaryContents', {})
                .get('sectionListRenderer', {})
                .get('contents', [])
            )
            
            for content in contents:
                items = content.get('itemSectionRenderer', {}).get('contents', [])
                
                for item in items[:max_results]:
                    video_renderer = item.get('videoRenderer', {})
                    if not video_renderer:
                        continue
                    
                    video_id = video_renderer.get('videoId', '')
                    if not video_id:
                        continue
                    
                    title_runs = video_renderer.get('title', {}).get('runs', [])
                    title = title_runs[0].get('text', 'No title') if title_runs else 'No title'
                    
                    channel_runs = video_renderer.get('ownerText', {}).get('runs', [])
                    channel = channel_runs[0].get('text', 'Unknown') if channel_runs else 'Unknown'
                    
                    view_count = video_renderer.get('viewCountText', {}).get('simpleText', 'N/A')
                    
                    videos.append({
                        'title': title,
                        'channel': channel,
                        'views': view_count,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })
                    
                    if len(videos) >= max_results:
                        break
                
                if len(videos) >= max_results:
                    break
            
        except Exception as e:
            logger.debug(f"HTML parsing error: {e}")
        
        return videos

class GitHubSearchConnector(BaseConnector):
    """
    Enhanced GitHub search with full descriptions and README fetching.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("github_search", timeout=20.0)
        self.base_url = "https://api.github.com/search/repositories"
        self.github_token = github_token
        
        if not github_token:
            logger.warning(
                "GitHub token not provided. API rate limits will be restrictive "
                "(60 requests/hour vs 5000 with token)."
            )
    
    async def _execute(self, params: Dict[str, Any]) -> ConnectorResponse:
        query = params.get("query")
        if not query:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error="'query' parameter is required."
            )
        
        include_readme = params.get("include_readme", True)
        max_results = min(params.get("max_results", 10), 20)
        
        search_params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": max_results
        }
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ResearchBot/2.0"
        }
        
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        # loop = asyncio.get_event_loop()
        
        try:
            response = await asyncio.to_thread(
                lambda: requests.get(
                    self.base_url, 
                    params=search_params, 
                    headers=headers, 
                    timeout=15
                )
            )
            response.raise_for_status()
            data = response.json()
            
            repos = data.get("items", [])
            if not repos:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data=f"No GitHub repositories found for '{query}'."
                )
            
            # Format results with enhanced information
            formatted_repos = []
            for i, repo in enumerate(repos, 1):
                name = repo.get("full_name", "Unknown")
                description = repo.get("description", "No description provided")
                
                # Truncate description if too long
                if len(description) > 1500:
                    description = description[:1500] + "..."
                
                stars = repo.get("stargazers_count", 0)
                forks = repo.get("forks_count", 0)
                language = repo.get("language", "Not specified")
                url = repo.get("html_url", "")
                updated = repo.get("updated_at", "Unknown")[:10]
                topics = repo.get("topics", [])
                license_info = repo.get("license", {})
                license_name = license_info.get("name", "No license") if license_info else "No license"
                
                repo_summary = (
                    f"{i}. {name}\n"
                    f"   Description: {description}\n"
                    f"   Language: {language} | Stars: {stars:,} | Forks: {forks:,}\n"
                    f"   License: {license_name} | Last updated: {updated}\n"
                )
                
                if topics:
                    repo_summary += f"   Topics: {', '.join(topics[:5])}\n"
                
                # Fetch README if requested
                if include_readme:
                    readme_content = await self._fetch_readme(
                        repo.get("full_name"), 
                        headers, 
                        # loop
                    )
                    if readme_content:
                        repo_summary += f"   README preview: {readme_content}\n"
                
                repo_summary += f"   URL: {url}"
                formatted_repos.append(repo_summary)
            
            result = f"GitHub repositories for '{query}':\n\n" + "\n\n".join(formatted_repos)
            
            return ConnectorResponse(
                success=True, 
                source=self.name, 
                data=result,
                metadata={
                    "total_count": data.get("total_count", 0),
                    "results_returned": len(repos),
                    "included_readmes": include_readme
                }
            )
            
        except requests.HTTPError as e:
            if e.response.status_code == 403:
                return ConnectorResponse(
                    success=False,
                    source=self.name,
                    error="GitHub API rate limit exceeded. Provide a GitHub token to increase limits."
                )
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"GitHub API error: {e.response.status_code}"
            )
        except Exception as e:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"GitHub search failed: {str(e)}"
            )
    
    async def _fetch_readme(
        self, 
        repo_full_name: str, 
        headers: Dict, 
        # loop
    ) -> str:
        """Fetch and extract README content."""
        try:
            readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
            
            response = await asyncio.to_thread(
                lambda: requests.get(readme_url, headers=headers, timeout=10)
            )
            
            if response.status_code == 404:
                return ""
            
            response.raise_for_status()
            readme_data = response.json()
            
            # README content is base64 encoded
            import base64
            content = base64.b64decode(readme_data.get("content", "")).decode('utf-8')
            
            # Extract first few meaningful lines (skip headers and badges)
            lines = content.split('\n')
            meaningful_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip common noise
                if (line and 
                    not line.startswith('#') and 
                    not line.startswith('![') and  # ✅ fixed invalid string
                    not line.startswith('[![') and
                    len(line) > 20):
                    meaningful_lines.append(line)
                    if len(meaningful_lines) >= 3:
                        break
            
            if meaningful_lines:
                preview = ' '.join(meaningful_lines)[:3000]
                return preview + "..."
            
            return ""
            
        except Exception as e:
            logger.debug(f"Could not fetch README for {repo_full_name}: {e}")
            return ""
   

class SafeWebScraperConnector:
    """Enhanced web scraper with comprehensive safety measures."""
    
    name: str = "safe_web_scraper"
    description: str = "A secure and robust web scraping tool with safety controls and content filtering."
    
    # Security configurations
    MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB limit
    MAX_REDIRECTS = 5
    REQUEST_TIMEOUT = 15
    MAX_SENTENCES = 200
    MIN_SENTENCE_LENGTH = 10
    MAX_SENTENCE_LENGTH = 1000
    
    # Blocked domains and patterns
    BLOCKED_DOMAINS = {
        "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
        "tiktok.com", "snapchat.com", 
        "login", "accounts.google", "microsoft.com", "apple.com",
        "paypal.com", "banking", "wallet", "crypto", "gambling"
    }
    
    BLOCKED_PATTERNS = [
        r"login|signin|signup|register|auth",
        r"admin|dashboard|control",
        r"private|internal|restricted",
        r"\.onion$",  # Tor hidden services
        r"localhost|127\.0\.0\.1|0\.0\.0\.0",
        r"192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\."  # Private IP ranges
    ]
    
    # Dangerous file extensions
    BLOCKED_EXTENSIONS = {
        ".exe", ".bat", ".cmd", ".scr", ".pif", ".com", ".jar",
        ".zip", ".rar", ".7z", ".tar", ".gz", ".dmg", ".pkg"
    }
    
    def __init__(self):
        self.ddg = DDGS()
        self._setup_session()
        
    def _setup_session(self):
        """Initialize secure HTTP session with safety configurations."""
        self.base_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",  # Do Not Track
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none"
        }
        
        # Legitimate user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]
        
        self._session = requests.Session()
        
        # Enhanced retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524],
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        
        # Set session-wide security settings
        self._session.max_redirects = self.MAX_REDIRECTS
        
    def _is_url_safe(self, url: str) -> tuple[bool, str]:
        """Comprehensive URL safety validation."""
        try:
            parsed = urlparse(url.lower())
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False, f"Unsafe scheme: {parsed.scheme}"
            
            # Prefer HTTPS
            if parsed.scheme == 'http':
                logger.warning(f"Using insecure HTTP for: {url}")
            
            # Check for blocked domains
            domain = parsed.netloc.lower()
            for blocked in self.BLOCKED_DOMAINS:
                if blocked in domain:
                    return False, f"Blocked domain pattern: {blocked}"
            
            # Check for dangerous patterns
            full_url = url.lower()
            for pattern in self.BLOCKED_PATTERNS:
                if re.search(pattern, full_url):
                    return False, f"Blocked URL pattern: {pattern}"
            
            # Check file extension
            path = parsed.path.lower()
            for ext in self.BLOCKED_EXTENSIONS:
                if path.endswith(ext):
                    return False, f"Blocked file extension: {ext}"
            
            # Check for suspicious ports
            if parsed.port and parsed.port not in [80, 443, 8080, 8443]:
                logger.warning(f"Unusual port {parsed.port} for URL: {url}")
            
            return True, "URL appears safe"
            
        except Exception as e:
            return False, f"URL parsing error: {str(e)}"
    
    def _search_articles(self, query: str, num_results: int = 3) -> List[str]:
        """Search for articles with enhanced filtering."""
        try:
            # Sanitize search query
            clean_query = re.sub(r'[^\w\s\-".]', '', query.strip())
            if not clean_query:
                return []
            
            results = self.ddg.text(clean_query, max_results=num_results * 3)  # Get more to filter
            safe_urls = []
            
            for result in results:
                url = result.get("href", "")
                if not url:
                    continue
                
                is_safe, reason = self._is_url_safe(url)
                if is_safe:
                    safe_urls.append(url)
                    if len(safe_urls) >= num_results:
                        break
                else:
                    logger.debug(f"Skipping URL {url}: {reason}")
            
            return safe_urls
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _validate_content_type(self, response: requests.Response) -> bool:
        """Validate response content type."""
        content_type = response.headers.get('content-type', '').lower()
        allowed_types = [
            'text/html', 'application/xhtml+xml', 'text/plain',
            'application/xml', 'text/xml'
        ]
        
        return any(allowed_type in content_type for allowed_type in allowed_types)
    
    def _clean_and_extract_text(self, html: str, url: str) -> List[str]:
        """Safely extract and clean text content."""
        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove potentially dangerous elements
            dangerous_tags = [
                "script", "style", "iframe", "object", "embed", "applet",
                "form", "input", "button", "textarea", "select",
                "link", "meta", "base", "noscript"
            ]
            
            for tag_name in dangerous_tags:
                for tag in soup.find_all(tag_name):
                    tag.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove navigation, ads, and other non-content elements
            non_content_selectors = [
                "nav", "footer", "header", "aside", "figure",
                ".advertisement", ".ad", ".ads", ".sidebar",
                ".navigation", ".menu", ".cookie", ".popup",
                "[class*='ad-']", "[id*='ad-']"
            ]
            
            for selector in non_content_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract text from paragraphs primarily
            content_elements = soup.find_all(["p", "article", "main", "div[class*='content']"])
            
            if not content_elements:
                # Fallback to all text if no paragraphs found
                content_elements = [soup]
            
            # Extract and clean sentences
            sentences = []
            for element in content_elements:
                text = element.get_text(" ", strip=True)
                if not text:
                    continue
                
                # Split into sentences more carefully
                sentence_parts = re.split(r'[.!?]+\s+', text)
                
                for sentence in sentence_parts:
                    sentence = sentence.strip()
                    
                    # Filter sentences
                    if (self.MIN_SENTENCE_LENGTH <= len(sentence) <= self.MAX_SENTENCE_LENGTH and
                        not re.search(r'^[\s\W]*$', sentence) and  # Not just whitespace/punctuation
                        not re.search(r'(cookie|privacy policy|terms of service)', sentence.lower())):
                        
                        sentences.append(sentence)
                        
                        if len(sentences) >= self.MAX_SENTENCES:
                            break
                
                if len(sentences) >= self.MAX_SENTENCES:
                    break
            
            return sentences[:self.MAX_SENTENCES]
            
        except Exception as e:
            logger.warning(f"Text extraction error for {url}: {e}")
            return []
    
    def _scrape_page(self, url: str) -> List[str]:
        """Safely scrape a single page with comprehensive error handling."""
        try:
            # Prepare headers
            headers = self.base_headers.copy()
            headers["User-Agent"] = random.choice(self.user_agents)
            headers["Referer"] = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
            
            # Primary scraping attempt
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT,
                    allow_redirects=True,
                    stream=True,
                    verify=True  # Keep this True for security
                )
                
                #  SUCCESS PATH - process the response
                response.raise_for_status()
                
                # Check content size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.MAX_CONTENT_SIZE:
                    logger.warning(f"Content too large: {content_length} bytes")
                    return []
                
                # Check content type
                if not self._validate_content_type(response):
                    logger.warning(f"Invalid content type: {response.headers.get('content-type')}")
                    return []
                
                # Read content with size limit
                content = b""
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > self.MAX_CONTENT_SIZE:
                        logger.warning("Content size limit exceeded during download")
                        return []
                
                html = content.decode('utf-8', errors='ignore')
                
            except requests.exceptions.SSLError as ssl_error:
                # Handle SSL errors specifically
                logger.warning(f"SSL verification failed for {url}: {ssl_error}")
                return []
                
            except (requests.RequestException, UnicodeDecodeError) as e:
                # FAILURE PATH - try CloudScraper fallback
                logger.info(f"Primary scraping failed, trying CloudScraper: {e}")
                
                try:
                    scraper = cloudscraper.create_scraper(
                        browser='chrome',
                        delay=1
                    )
                    response = scraper.get(url, timeout=self.REQUEST_TIMEOUT, verify=True)
                    response.raise_for_status()
                    html = response.text
                    
                except requests.exceptions.SSLError as ssl_error:
                    logger.warning(f"CloudScraper SSL verification failed for {url}: {ssl_error}")
                    return []
                    
                except Exception as fallback_error:
                    logger.warning(f"CloudScraper also failed for {url}: {fallback_error}")
                    return []
            
            # Extract and clean content (both paths lead here if successful)
            sentences = self._clean_and_extract_text(html, url)
            
            # Rate limiting
            time.sleep(random.uniform(1.0, 2.0))
            
            return sentences
            
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            return []
    
    async def __call__(self, params: Dict[str, Any]) -> ConnectorResponse:
        """Main scraping function with comprehensive error handling."""
        query = params.get("query", "").strip()
        num_results = min(params.get("num_results", 1), 5)  # Limit max results
        
        if not query:
            return ConnectorResponse(
                success=False, 
                source=self.name, 
                error="Query parameter is required and cannot be empty."
            )
        
        # Validate query
        if len(query) > 500:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="Query too long (max 500 characters)."
            )
        
        try:
            # Search for articles
            urls = await asyncio.to_thread(self._search_articles, query, num_results)
            
            if not urls:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data="No safe articles found for the given query."
                )
            
            # Scrape the first available URL
            all_content = []
            for url in urls:
                logger.info(f"Scraping: {url}")
                
                scraped_sentences = await asyncio.to_thread(self._scrape_page, url)
                
                if scraped_sentences:
                    all_content.extend(scraped_sentences)
                    break  # Success with first URL
                else:
                    logger.warning(f"No content extracted from: {url}")
            
            if not all_content:
                return ConnectorResponse(
                    success=True,
                    source=self.name,
                    data="Unable to extract content from any of the found articles."
                )
            
            # Join content and add metadata
            content = "\n".join(all_content)
            final_data = f"Content extracted from: {urls[0] if urls else 'Unknown'}\n\n{content}"
            
            return ConnectorResponse(
                success=True,
                source=self.name,
                data=final_data
            )
            
        except asyncio.TimeoutError:
            return ConnectorResponse(
                success=False,
                source=self.name,
                error="Request timed out."
            )
        except Exception as e:
            logger.error(f"Unexpected error in web scraper: {e}")
            return ConnectorResponse(
                success=False,
                source=self.name,
                error=f"An unexpected error occurred: {str(e)[:100]}"
            )

# --- Main MCP Connector Manager ---
class MCPManager:
    """Manages all available tool connectors and orchestrates their execution."""
    
    def __init__(self, llm: Any, tavily_api_key: Optional[str] = None):
        self.connectors: Dict[str, BaseConnector] = {}
        self._register_default_connectors(llm, tavily_api_key)

    def _register_default_connectors(self, llm: Any, tavily_api_key: Optional[str]):
        """Initializes and registers the default set of tools."""
        if tavily_api_key:
            self.register_connector(TavilySearchConnector(api_key=tavily_api_key))
        else:
            self.register_connector(DuckDuckGoSearchConnector())
            
        self.register_connector(WikipediaConnector())
        self.register_connector(MathConnector(llm=llm))
        self.register_connector(YahooFinanceConnector())
        self.register_connector(ArXivConnector())
        self.register_connector(SymPySolverConnector())
        self.register_connector(SafeWebScraperConnector())
        
        # --- Research and academic connectors (merged here) ---
        self.register_connector(SemanticScholarConnector())
        self.register_connector(PubMedConnector())
        self.register_connector(RedditSearchConnector())
        YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        self.register_connector(YouTubeSearchConnector(api_key=YOUTUBE_API_KEY))

        self.register_connector(BrightDataRedditSearchConnector())
        self.register_connector(BrightDataRedditPostRetrievalConnector())
        logger.info("Registered Bright Data connectors for deep research.")

        github_token = os.getenv("GITHUB_TOKEN")  # Optional
        self.register_connector(GitHubSearchConnector(github_token=github_token))
        logger.info("Registered 5 additional research connectors")
        # ------------------------------------------------------

        if os.getenv("OPENWEATHERMAP_API_KEY"):
            self.register_connector(WeatherConnector(api_key=os.getenv("OPENWEATHERMAP_API_KEY")))
        else:
            logger.warning("OPENWEATHERMAP_API_KEY not found in environment variables. Weather tool will be disabled.")
            
        if os.getenv("NEWS_API_KEY"):
            self.register_connector(NewsConnector(api_key=os.getenv("NEWS_API_KEY")))
        else:
            logger.warning("NEWS_API_KEY not found in environment variables. News tool will be disabled.")

    def register_connector(self, connector: BaseConnector):
        """Registers a new connector instance."""
        logger.info(f"Registering MCP connector: {connector.name}")
        self.connectors[connector.name] = connector

    def get_primary_web_search_name(self) -> str:
        """Returns the name of the primary registered web search connector."""
        return "tavily_web_search" if "tavily_web_search" in self.connectors else "duckduckgo_web_search"
    
    def get_available_connectors(self) -> List[str]:
        """Return list of all available connector names."""
        return list(self.connectors.keys())
    
    def get_connector(self, name: str) -> Optional[BaseConnector]:
        """Get a specific connector by name."""
        return self.connectors.get(name)

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ConnectorResponse:
        """Executes a single tool call."""
        if tool_name in self.connectors:
            return await self.connectors[tool_name](params)
        else:
            logger.warning(f"Attempted to call unregistered tool: {tool_name}")
            return ConnectorResponse(success=False, source=tool_name, error="Tool not found.")
    
    def invoke(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Synchronous wrapper for tool execution (backwards compatibility)."""
        return asyncio.run(self.execute_tool(tool_name, params))

    def get_system_health(self) -> Dict[str, Any]:
        """Returns a report on the health of all registered connectors."""
        return {
            "total_connectors": len(self.connectors),
            "available_connectors": list(self.connectors.keys()),
            "connector_health": {name: conn.get_health() for name, conn in self.connectors.items()}
        }
