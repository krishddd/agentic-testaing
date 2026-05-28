"""
Screenshot Capture Tools

Capture desktop screenshots using mss (fast) with Pillow fallback.
Supports full screen, specific regions, and window capture.
"""

import os
import logging
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Graceful imports
try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from PIL import ImageGrab, Image
    HAS_PILLOW_GRAB = True
except ImportError:
    HAS_PILLOW_GRAB = False

if not HAS_MSS and not HAS_PILLOW_GRAB:
    logger.warning("[ScreenshotTools] Neither mss nor Pillow installed — screenshots disabled")


@dataclass
class ScreenshotResult:
    """Result of a screenshot operation."""
    success: bool
    operation: str
    message: str
    path: Optional[str] = None
    data: Optional[Dict] = None
    error: Optional[str] = None


class ScreenshotTools:
    """
    Desktop screenshot capture tools.
    
    Capabilities:
    - Full screen capture
    - Region capture (left, top, width, height)
    - Auto-named screenshots with timestamps
    """

    def __init__(self, default_dir: str = "."):
        self.default_dir = default_dir

    def _generate_path(self, prefix: str = "screenshot") -> str:
        """Generate a timestamped screenshot filename."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.default_dir, f"{prefix}_{ts}.png")

    def capture_screen(self, output_path: Optional[str] = None) -> ScreenshotResult:
        """
        Capture the full screen.
        """
        out = output_path or self._generate_path("fullscreen")
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)

        # Method 1: mss (fastest)
        if HAS_MSS:
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[0]  # Full virtual screen
                    screenshot = sct.grab(monitor)
                    mss.tools.to_png(screenshot.rgb, screenshot.size, output=out)

                file_size = os.path.getsize(out)
                return ScreenshotResult(
                    success=True, operation="capture_screen",
                    message=f"Screenshot saved: {out} ({monitor['width']}x{monitor['height']})",
                    path=out,
                    data={
                        "width": monitor['width'],
                        "height": monitor['height'],
                        "size_bytes": file_size,
                        "method": "mss",
                    }
                )
            except Exception as e:
                logger.warning(f"[ScreenshotTools] mss failed: {e}, trying Pillow fallback")

        # Method 2: Pillow ImageGrab (fallback)
        if HAS_PILLOW_GRAB:
            try:
                screenshot = ImageGrab.grab()
                screenshot.save(out)

                file_size = os.path.getsize(out)
                return ScreenshotResult(
                    success=True, operation="capture_screen",
                    message=f"Screenshot saved: {out} ({screenshot.width}x{screenshot.height})",
                    path=out,
                    data={
                        "width": screenshot.width,
                        "height": screenshot.height,
                        "size_bytes": file_size,
                        "method": "pillow",
                    }
                )
            except Exception as e:
                return ScreenshotResult(
                    success=False, operation="capture_screen", message="",
                    error=f"Pillow screenshot failed: {str(e)}"
                )

        return ScreenshotResult(
            success=False, operation="capture_screen", message="",
            error="No screenshot library available. Install: pip install mss"
        )

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        output_path: Optional[str] = None,
    ) -> ScreenshotResult:
        """
        Capture a specific region of the screen.
        """
        out = output_path or self._generate_path("region")
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)

        # Method 1: mss
        if HAS_MSS:
            try:
                with mss.mss() as sct:
                    region = {"left": left, "top": top, "width": width, "height": height}
                    screenshot = sct.grab(region)
                    mss.tools.to_png(screenshot.rgb, screenshot.size, output=out)

                file_size = os.path.getsize(out)
                return ScreenshotResult(
                    success=True, operation="capture_region",
                    message=f"Region captured: ({left},{top}) {width}x{height}",
                    path=out,
                    data={
                        "region": [left, top, width, height],
                        "size_bytes": file_size,
                        "method": "mss",
                    }
                )
            except Exception as e:
                logger.warning(f"[ScreenshotTools] mss region failed: {e}")

        # Method 2: Pillow
        if HAS_PILLOW_GRAB:
            try:
                screenshot = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                screenshot.save(out)

                file_size = os.path.getsize(out)
                return ScreenshotResult(
                    success=True, operation="capture_region",
                    message=f"Region captured: ({left},{top}) {width}x{height}",
                    path=out,
                    data={
                        "region": [left, top, width, height],
                        "size_bytes": file_size,
                        "method": "pillow",
                    }
                )
            except Exception as e:
                return ScreenshotResult(
                    success=False, operation="capture_region", message="",
                    error=f"Region capture failed: {str(e)}"
                )

        return ScreenshotResult(
            success=False, operation="capture_region", message="",
            error="No screenshot library available"
        )

    def capture_window(
        self,
        window_title: str,
        output_path: Optional[str] = None,
    ) -> ScreenshotResult:
        """
        Capture a specific window by title (Windows only).
        Falls back to full screen if window not found.
        """
        out = output_path or self._generate_path("window")

        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Find window by title
            hwnd = user32.FindWindowW(None, window_title)
            if not hwnd:
                # Try partial match
                import win32gui
                def callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if window_title.lower() in title.lower():
                            results.append((hwnd, title))
                
                results = []
                try:
                    win32gui.EnumWindows(callback, results)
                except Exception:
                    pass
                
                if results:
                    hwnd = results[0][0]
                else:
                    return self.capture_screen(out)  # Fallback to full screen

            # Get window rect
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            
            left = rect.left
            top = rect.top
            width = rect.right - rect.left
            height = rect.bottom - rect.top

            return self.capture_region(left, top, width, height, out)

        except ImportError:
            logger.warning("[ScreenshotTools] win32gui not available, falling back to full screen")
            return self.capture_screen(out)
        except Exception as e:
            return ScreenshotResult(
                success=False, operation="capture_window", message="",
                error=f"Window capture failed: {str(e)}"
            )
