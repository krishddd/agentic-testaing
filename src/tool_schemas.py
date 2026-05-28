from pydantic import BaseModel, Field
from typing import Optional

class WebSearchInput(BaseModel):
    """Input for a web search."""
    query: str = Field(description="The search query.")

class TextAnalysisInput(BaseModel):
    """Input for text analysis tool."""
    text: str = Field(description="The text to analyze for word count, readability, etc.")

class WikipediaInput(BaseModel):
    """Input for a Wikipedia search."""
    query: str = Field(description="The query for Wikipedia.")

class CalculatorInput(BaseModel):
    """Input for the calculator."""
    expression: str = Field(description="The mathematical expression to evaluate.")

class SymPySolverInput(BaseModel):
    equation: str = Field(..., description="The algebraic equation to solve, e.g., '2*x + 10 = 20'.")

class WebScraperInput(BaseModel):
    """Input for the web scraper tool."""
    query: str = Field(description="The topic to search for and scrape a single article from.")
    
class SemanticScholarInput(BaseModel):
    query: str = Field(description="The search query for academic papers on Semantic Scholar.")

class PubMedInput(BaseModel):
    query: str = Field(description="The search query for biomedical literature on PubMed.")

class RedditSearchInput(BaseModel):
    query: str = Field(description="The search query for discussions on Reddit.")

class YouTubeSearchInput(BaseModel):
    query: str = Field(description="The search query for videos on YouTube.")

class GitHubSearchInput(BaseModel):
    query: str = Field(description="The search query for code repositories on GitHub.")

class DateTimeInput(BaseModel):
    query: Optional[str] = Field(description="A specific query about the date or time, e.g., 'current time'. If empty, returns full date and time.")

class StockTickerInput(BaseModel):
    """Input for fetching stock prices."""
    ticker: str = Field(description="The stock ticker symbol (e.g., 'AAPL', 'GOOGL').")

class WeatherInput(BaseModel):
    """Input for fetching the weather forecast."""
    city: str = Field(description="The city for the weather forecast (e.g., 'London', 'Tokyo').")

class NewsInput(BaseModel):
    """Input for fetching news headlines."""
    query: str = Field(description="The topic to search for in the news.")

class ArXivInput(BaseModel):
    """Input for searching academic papers on ArXiv."""
    query: str = Field(description="The query for searching academic papers.")

class FinalAnswerInput(BaseModel):
    """Input for the final answer tool."""
    summary: str = Field(description="The final, consolidated answer for the user.")
