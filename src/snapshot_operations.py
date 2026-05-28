import time
import requests
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def poll_snapshot_status(snapshot_id: str, max_attempts: int = 60, poll_interval: int = 5) -> bool:
    """
    Polls the Bright Data API to check if a snapshot is ready for download.
    
    Args:
        snapshot_id: The unique identifier for the snapshot
        max_attempts: Maximum number of polling attempts (default: 60)
        poll_interval: Seconds to wait between polls (default: 5)
    
    Returns:
        True if snapshot is ready, False if it failed or timed out
    """
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        logger.error("BRIGHTDATA_API_KEY not found in environment variables")
        return False
    
    # Remove any placeholder prefixes that were added for tracking
    actual_snapshot_id = snapshot_id.replace("reddit_search_", "").replace("reddit_comments_", "")
    
    status_url = f"https://api.brightdata.com/datasets/v3/snapshot/{actual_snapshot_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    logger.info(f"Starting to poll snapshot status for: {actual_snapshot_id}")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(status_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            status_data = response.json()
            status = status_data.get("status", "unknown")
            
            logger.info(f"Poll attempt {attempt}/{max_attempts} - Snapshot {actual_snapshot_id} status: {status}")
            
            if status == "ready":
                logger.info(f"Snapshot {actual_snapshot_id} is ready for download")
                return True
            elif status in ["failed", "error"]:
                error_msg = status_data.get("error", "Unknown error")
                logger.error(f"Snapshot {actual_snapshot_id} failed: {error_msg}")
                return False
            elif status in ["running", "pending"]:
                # Still processing, continue polling
                time.sleep(poll_interval)
            else:
                logger.warning(f"Unknown snapshot status: {status}")
                time.sleep(poll_interval)
                
        except requests.RequestException as e:
            logger.error(f"Error polling snapshot status (attempt {attempt}): {e}")
            if attempt < max_attempts:
                time.sleep(poll_interval)
            else:
                return False
        except Exception as e:
            logger.error(f"Unexpected error polling snapshot (attempt {attempt}): {e}")
            return False
    
    logger.error(f"Snapshot {actual_snapshot_id} timed out after {max_attempts} attempts")
    return False


def download_snapshot(snapshot_id: str) -> Optional[Any]:
    """
    Downloads the snapshot data from Bright Data API and returns the parsed JSON.
    
    Args:
        snapshot_id: The unique identifier for the snapshot (may have prefix for tracking)
    
    Returns:
        Parsed JSON data from the snapshot, or None if download failed
    """
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        logger.error("BRIGHTDATA_API_KEY not found in environment variables")
        return None
    
    # Store the original ID to determine data type
    original_snapshot_id = snapshot_id
    
    # Remove any placeholder prefixes for API call
    actual_snapshot_id = snapshot_id.replace("reddit_search_", "").replace("reddit_comments_", "")
    
    download_url = f"https://api.brightdata.com/datasets/v3/snapshot/{actual_snapshot_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    logger.info(f"Downloading snapshot data for: {actual_snapshot_id}")
    
    try:
        response = requests.get(download_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Log download success with data size info
        if isinstance(data, list):
            logger.info(f"Successfully downloaded snapshot {actual_snapshot_id} - {len(data)} records")
        else:
            logger.info(f"Successfully downloaded snapshot {actual_snapshot_id}")
        
        # Validate data structure based on snapshot type
        if "reddit_search" in original_snapshot_id:
            # Validate Reddit search data
            if isinstance(data, list) and len(data) > 0:
                # Check if first item has expected fields
                if not all(key in data[0] for key in ["title", "url"]):
                    logger.warning("Reddit search data missing expected fields (title, url)")
            else:
                logger.warning("Reddit search returned empty or invalid data")
                
        elif "reddit_comments" in original_snapshot_id:
            # Validate Reddit comments data
            if isinstance(data, list) and len(data) > 0:
                # Check if first item has expected fields
                if not all(key in data[0] for key in ["comment_id", "comment"]):
                    logger.warning("Reddit comments data missing expected fields (comment_id, comment)")
            else:
                logger.warning("Reddit comments returned empty or invalid data")
        
        return data
        
    except requests.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 404:
            logger.error(f"Snapshot {actual_snapshot_id} not found (404)")
        elif status_code == 403:
            logger.error(f"Access forbidden for snapshot {actual_snapshot_id} (403) - check API key permissions")
        elif status_code == 429:
            logger.error(f"Rate limit exceeded when downloading snapshot {actual_snapshot_id}")
        else:
            logger.error(f"HTTP error downloading snapshot {actual_snapshot_id}: {status_code}")
        
        # Try to get error details from response
        try:
            error_data = e.response.json()
            logger.error(f"Error details: {error_data}")
        except:
            pass
            
        return None
        
    except requests.RequestException as e:
        logger.error(f"Network error downloading snapshot {actual_snapshot_id}: {e}")
        return None
        
    except ValueError as e:
        logger.error(f"Failed to parse JSON response for snapshot {actual_snapshot_id}: {e}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error downloading snapshot {actual_snapshot_id}: {e}", exc_info=True)
        return None


def get_snapshot_info(snapshot_id: str) -> Optional[dict]:
    """
    Get detailed information about a snapshot without downloading the full data.
    
    Args:
        snapshot_id: The unique identifier for the snapshot
    
    Returns:
        Dictionary with snapshot metadata, or None if request failed
    """
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        logger.error("BRIGHTDATA_API_KEY not found in environment variables")
        return None
    
    actual_snapshot_id = snapshot_id.replace("reddit_search_", "").replace("reddit_comments_", "")
    
    info_url = f"https://api.brightdata.com/datasets/v3/snapshot/{actual_snapshot_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(info_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        info = response.json()
        logger.info(f"Retrieved info for snapshot {actual_snapshot_id}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get snapshot info for {actual_snapshot_id}: {e}")
        return None
# import time
# import requests
# import os

# def poll_snapshot_status(snapshot_id: str) -> bool:
#     """
#     NOTE: This is a placeholder implementation.
#     The actual implementation will depend on the Bright Data API specifics for polling.
#     This version will simulate a short wait and always return True.
#     """
#     print(f"Polling status for snapshot_id: {snapshot_id} (placeholder)")
#     time.sleep(2) # Simulate polling delay
#     print("Snapshot is ready (placeholder).")
#     return True

# def download_snapshot(snapshot_id: str):
#     """
#     NOTE: This is a placeholder implementation.
#     The actual implementation should handle downloading and parsing the snapshot data
#     from the Bright Data API. This version returns dummy data.
#     """
#     print(f"Downloading snapshot for snapshot_id: {snapshot_id} (placeholder)")
#     # In a real scenario, you would make a request to the Bright Data API to get the snapshot data.
#     # For example:
#     api_key = os.getenv("BRIGHTDATA_API_KEY")
#     headers = {"Authorization": f"Bearer {api_key}"}
#     response = requests.get(f"https://api.brightdata.com/snapshots/v1/{snapshot_id}", headers=headers)
#     response.raise_for_status()
#     return response.json()
    
#     # Returning dummy data for demonstration
#     # if "reddit_search" in snapshot_id: # A simple way to distinguish for dummy data
#     #     return [
#     #         {"title": "Dummy Reddit Post 1", "url": "https://www.reddit.com/r/dummy/comments/1"},
#     #         {"title": "Dummy Reddit Post 2", "url": "https://www.reddit.com/r/dummy/comments/2"},
#     #     ]
#     # elif "reddit_comments" in snapshot_id:
#     #     return [
#     #         {"comment_id": "c1", "comment": "This is a dummy comment.", "date_posted": "2024-01-01"},
#     #         {"comment_id": "c2", "comment": "This is another dummy comment.", "date_posted": "2024-01-02"},
#     #     ]
#     # return []
