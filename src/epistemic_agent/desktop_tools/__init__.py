"""
Desktop Tools — Real-world tool executors for the Epistemic Agent.

Provides real file system operations, code interpretation,
system monitoring, clipboard, browser, PDF, image, and screenshot capabilities.

Architecture (inspired by Open Interpreter + OS-Copilot):
  - FileSystemExecutor: 10 real file operations with audit trail
  - CodeInterpreter: Sandboxed Python/Shell execution
  - SandboxManager: Safety checks + process isolation
  - SystemMonitor: CPU, RAM, disk, processes
  - ClipboardTools: Read/write clipboard
  - BrowserTools: Chrome automation via Selenium
  - PDFTools: Create/read/merge PDFs
  - ImageTools: Create/edit/convert images
  - ScreenshotTools: Desktop/region/window capture
"""

from .file_ops import FileSystemExecutor, FileResult
from .sandbox import SandboxManager, ExecutionResult, ExecutionStatus
from .code_interpreter import CodeInterpreter
from .system_monitor import SystemMonitor, SystemInfo, ProcessInfo
from .clipboard_tools import ClipboardTools
from .browser_tools import BrowserTools, BrowserResult
from .pdf_tools import PDFTools, PDFResult
from .image_tools import ImageTools, ImageResult
from .screenshot_tools import ScreenshotTools, ScreenshotResult
