"""
Clipboard Tools — Cross-platform clipboard access.

Uses pyperclip for clipboard read/write operations.
"""

import logging
from typing import Optional

try:
    import pyperclip
except ImportError:
    pyperclip = None

logger = logging.getLogger(__name__)


class ClipboardTools:
    """Cross-platform clipboard access."""

    @staticmethod
    def get_clipboard() -> str:
        """Get current clipboard contents."""
        if not pyperclip:
            return "[Error: pyperclip not installed]"
        try:
            return pyperclip.paste() or "(clipboard is empty)"
        except Exception as e:
            return f"[Error reading clipboard: {e}]"

    @staticmethod
    def set_clipboard(text: str) -> bool:
        """Copy text to clipboard."""
        if not pyperclip:
            logger.error("pyperclip not installed")
            return False
        try:
            pyperclip.copy(text)
            return True
        except Exception as e:
            logger.error(f"Clipboard write failed: {e}")
            return False
