"""
Browser Automation Tools

Chrome browser control via Selenium for:
  - Opening URLs and extracting page content
  - Google search with result extraction
  - Full-page screenshots
  - Form filling and element clicking
  
Uses headless Chrome by default. Graceful fallback if
selenium/chromedriver not installed.
"""

import os
import re
import logging
import time
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Graceful imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    logger.warning("[BrowserTools] selenium not installed — browser tools disabled")


@dataclass
class BrowserResult:
    """Result of a browser operation."""
    success: bool
    operation: str
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None


class BrowserTools:
    """
    Chrome browser automation for the epistemic agent.
    
    Provides headless browser capabilities:
    - Navigate to URLs and extract text
    - Google search with structured results
    - Page screenshots
    - Form interaction
    """

    def __init__(self, headless: bool = True, timeout: int = 15):
        self.headless = headless
        self.timeout = timeout
        self._driver = None
        self._available = HAS_SELENIUM

    def _get_driver(self):
        """Initialize or return existing Chrome driver."""
        if not self._available:
            return None
        
        if self._driver is None:
            try:
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--disable-extensions")
                options.add_argument("--disable-popup-blocking")
                options.add_argument("--log-level=3")
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                
                self._driver = webdriver.Chrome(options=options)
                self._driver.set_page_load_timeout(self.timeout)
                logger.info("[BrowserTools] Chrome driver initialized")
            except Exception as e:
                logger.error(f"[BrowserTools] Failed to init Chrome: {e}")
                self._available = False
                return None
        
        return self._driver

    def open_url(self, url: str) -> BrowserResult:
        """
        Open a URL and return page title + extracted text.
        """
        if not self._available:
            return BrowserResult(
                success=False, operation="open_url", message="",
                error="Selenium not installed. Install: pip install selenium"
            )

        driver = self._get_driver()
        if not driver:
            return BrowserResult(
                success=False, operation="open_url", message="",
                error="Chrome driver unavailable"
            )

        try:
            # Add protocol if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            driver.get(url)
            time.sleep(2)  # Allow page to load

            title = driver.title or "No title"
            
            # Extract main text content
            try:
                body = driver.find_element(By.TAG_NAME, "body")
                text = body.text[:3000]  # Limit text length
            except Exception:
                text = "Could not extract page text"

            return BrowserResult(
                success=True, operation="open_url",
                message=f"Opened: {title}",
                data={
                    "url": driver.current_url,
                    "title": title,
                    "text_preview": text[:500],
                    "full_text_length": len(text),
                }
            )
        except Exception as e:
            return BrowserResult(
                success=False, operation="open_url", message="",
                error=f"Failed to open {url}: {str(e)}"
            )

    def search_google(self, query: str, num_results: int = 5) -> BrowserResult:
        """
        Search Google and return structured results.
        """
        if not self._available:
            return BrowserResult(
                success=False, operation="search_google", message="",
                error="Selenium not installed"
            )

        driver = self._get_driver()
        if not driver:
            return BrowserResult(
                success=False, operation="search_google", message="",
                error="Chrome driver unavailable"
            )

        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            driver.get(search_url)
            time.sleep(2)

            results = []
            try:
                # Extract search results
                search_results = driver.find_elements(By.CSS_SELECTOR, "div.g")
                for i, result in enumerate(search_results[:num_results]):
                    try:
                        title_el = result.find_element(By.CSS_SELECTOR, "h3")
                        link_el = result.find_element(By.CSS_SELECTOR, "a")
                        snippet_el = result.find_elements(By.CSS_SELECTOR, "div[data-sncf], span.st, div.VwiC3b")
                        
                        results.append({
                            "position": i + 1,
                            "title": title_el.text if title_el else "No title",
                            "url": link_el.get_attribute("href") if link_el else "",
                            "snippet": snippet_el[0].text if snippet_el else "No snippet",
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            if not results:
                # Fallback: get all text
                body_text = driver.find_element(By.TAG_NAME, "body").text[:2000]
                return BrowserResult(
                    success=True, operation="search_google",
                    message=f"Search results for: {query}",
                    data={"query": query, "raw_text": body_text, "results": []}
                )

            return BrowserResult(
                success=True, operation="search_google",
                message=f"Found {len(results)} results for: {query}",
                data={"query": query, "results": results}
            )
        except Exception as e:
            return BrowserResult(
                success=False, operation="search_google", message="",
                error=f"Search failed: {str(e)}"
            )

    def screenshot_page(self, url: str, save_path: str) -> BrowserResult:
        """
        Take a screenshot of a web page.
        """
        if not self._available:
            return BrowserResult(
                success=False, operation="screenshot_page", message="",
                error="Selenium not installed"
            )

        driver = self._get_driver()
        if not driver:
            return BrowserResult(
                success=False, operation="screenshot_page", message="",
                error="Chrome driver unavailable"
            )

        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            driver.get(url)
            time.sleep(3)

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            
            driver.save_screenshot(save_path)

            return BrowserResult(
                success=True, operation="screenshot_page",
                message=f"Screenshot saved: {save_path}",
                data={"url": url, "path": save_path, "title": driver.title}
            )
        except Exception as e:
            return BrowserResult(
                success=False, operation="screenshot_page", message="",
                error=f"Screenshot failed: {str(e)}"
            )

    def get_page_text(self, url: str) -> BrowserResult:
        """
        Extract all text content from a web page.
        """
        if not self._available:
            return BrowserResult(
                success=False, operation="get_page_text", message="",
                error="Selenium not installed"
            )

        driver = self._get_driver()
        if not driver:
            return BrowserResult(
                success=False, operation="get_page_text", message="",
                error="Chrome driver unavailable"
            )

        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            driver.get(url)
            time.sleep(2)

            body = driver.find_element(By.TAG_NAME, "body")
            text = body.text

            return BrowserResult(
                success=True, operation="get_page_text",
                message=f"Extracted {len(text)} characters from {driver.title}",
                data={
                    "url": driver.current_url,
                    "title": driver.title,
                    "text": text[:5000],
                    "total_length": len(text),
                }
            )
        except Exception as e:
            return BrowserResult(
                success=False, operation="get_page_text", message="",
                error=f"Text extraction failed: {str(e)}"
            )

    def fill_form(self, selector: str, text: str) -> BrowserResult:
        """
        Fill a form field identified by CSS selector.
        Requires a page to already be open.
        """
        if not self._available or not self._driver:
            return BrowserResult(
                success=False, operation="fill_form", message="",
                error="No page open or Selenium not installed"
            )

        try:
            element = WebDriverWait(self._driver, self.timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            element.clear()
            element.send_keys(text)

            return BrowserResult(
                success=True, operation="fill_form",
                message=f"Filled '{selector}' with text",
                data={"selector": selector, "text_length": len(text)}
            )
        except Exception as e:
            return BrowserResult(
                success=False, operation="fill_form", message="",
                error=f"Fill failed: {str(e)}"
            )

    def click_element(self, selector: str) -> BrowserResult:
        """
        Click an element identified by CSS selector.
        Requires a page to already be open.
        """
        if not self._available or not self._driver:
            return BrowserResult(
                success=False, operation="click_element", message="",
                error="No page open or Selenium not installed"
            )

        try:
            element = WebDriverWait(self._driver, self.timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            element.click()
            time.sleep(1)

            return BrowserResult(
                success=True, operation="click_element",
                message=f"Clicked '{selector}'",
                data={"selector": selector, "current_url": self._driver.current_url}
            )
        except Exception as e:
            return BrowserResult(
                success=False, operation="click_element", message="",
                error=f"Click failed: {str(e)}"
            )

    def close(self):
        """Close the browser driver."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def __del__(self):
        self.close()
