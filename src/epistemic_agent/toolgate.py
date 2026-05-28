from typing import Callable, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import glob

# Fix 2: Guard tavily import to prevent crash if not installed
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from .generative_model import BeliefState, Action, ActionType, RiskLevel, FileStatus
from .config import settings
from .desktop_tools.file_ops import FileSystemExecutor
from .desktop_tools.code_interpreter import CodeInterpreter
from .desktop_tools.system_monitor import SystemMonitor
from .desktop_tools.clipboard_tools import ClipboardTools
from .desktop_tools.browser_tools import BrowserTools
from .desktop_tools.pdf_tools import PDFTools
from .desktop_tools.image_tools import ImageTools
from .desktop_tools.screenshot_tools import ScreenshotTools

@dataclass
class ToolContract:
    name: str
    preconditions: List[Callable[[BeliefState], bool]]
    postconditions: List[Callable[[str, BeliefState], bool]] # observation, current_belief
    risk_level: RiskLevel
    description: str

class ToolGate:
    """
    Executes tools ONLY if contracts are satisfied.
    Acts as the boundary between the agent and the real world.
    """
    
    def __init__(self):
        self.tavily_client = None
        if TavilyClient and settings.TAVILY_API_KEY:
            self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY.get_secret_value())
        
        # Define Contracts
        self.contracts = {
            "list_files": ToolContract(
                name="list_files",
                preconditions=[],
                postconditions=[lambda obs, b: "files found" in obs.lower() or "no_results" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="List files in current directory to verify existence"
            ),
            "web_search": ToolContract(
                name="web_search",
                preconditions=[],
                postconditions=[lambda obs, b: "success" in obs.lower() or "error" in obs.lower() or "no_results" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Search the web for verifying facts"
            ),
            "delete_file": ToolContract(
                name="delete_file",
                preconditions=[
                    lambda b: b.file_status_probs.get(FileStatus.EXISTS, 0) > 0.90
                ],
                postconditions=[lambda obs, b: "deleted" in obs.lower() or "trash" in obs.lower() or "error" in obs.lower()],
                risk_level=RiskLevel.HAZARDOUS,
                description="Delete a file (moves to trash by default)"
            ),
            "ask_user": ToolContract(
                name="ask_user",
                preconditions=[],
                postconditions=[lambda obs, b: "clarification_request" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Ask user for clarification"
            ),
            "read_file": ToolContract(
                name="read_file",
                preconditions=[
                    lambda b: b.file_status_probs.get(FileStatus.EXISTS, 0) > 0.5
                ],
                postconditions=[lambda obs, b: "contents" in obs.lower() or "error" in obs.lower() or "read" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Read contents of a file"
            ),
            "answer_user": ToolContract(
                name="answer_user",
                preconditions=[],  # EFE scoring gates this via entropy + evidence count
                postconditions=[lambda obs, b: "final_answer" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Provide final answer to user"
            ),
            # --- New Desktop Tools ------------------------
            "move_file": ToolContract(
                name="move_file",
                preconditions=[
                    lambda b: b.file_status_probs.get(FileStatus.EXISTS, 0) > 0.7
                ],
                postconditions=[lambda obs, b: "moved" in obs.lower() or "error" in obs.lower()],
                risk_level=RiskLevel.MODERATE,
                description="Move or rename a file"
            ),
            "copy_file": ToolContract(
                name="copy_file",
                preconditions=[
                    lambda b: b.file_status_probs.get(FileStatus.EXISTS, 0) > 0.5
                ],
                postconditions=[lambda obs, b: "copied" in obs.lower() or "error" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Copy a file or directory"
            ),
            "write_file": ToolContract(
                name="write_file",
                preconditions=[],
                postconditions=[lambda obs, b: "created" in obs.lower() or "overwrote" in obs.lower() or "error" in obs.lower()],
                risk_level=RiskLevel.MODERATE,
                description="Create or overwrite a file"
            ),
            "get_file_info": ToolContract(
                name="get_file_info",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Get file size, dates, and metadata"
            ),
            "search_files": ToolContract(
                name="search_files",
                preconditions=[],
                postconditions=[lambda obs, b: "found" in obs.lower() or "error" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Search files by name or content"
            ),
            "create_directory": ToolContract(
                name="create_directory",
                preconditions=[],
                postconditions=[lambda obs, b: "created" in obs.lower() or "exists" in obs.lower() or "error" in obs.lower()],
                risk_level=RiskLevel.SAFE,
                description="Create a new directory"
            ),
            "get_directory_tree": ToolContract(
                name="get_directory_tree",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Show directory structure as a tree"
            ),
        }
        
        # --- Code Execution + System Tools ----------------
        self.contracts.update({
            "run_code": ToolContract(
                name="run_code",
                preconditions=[],  # Sandbox handles safety
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.MODERATE,
                description="Execute Python code in sandbox"
            ),
            "run_shell": ToolContract(
                name="run_shell",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.MODERATE,
                description="Execute shell command in sandbox"
            ),
            "system_info": ToolContract(
                name="system_info",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Get system information (CPU, RAM, disk)"
            ),
            "get_processes": ToolContract(
                name="get_processes",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="List running processes"
            ),
            "clipboard_read": ToolContract(
                name="clipboard_read",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Read clipboard contents"
            ),
        })
        
        # --- Browser, PDF, Image, Screenshot Tools ----------------
        self.contracts.update({
            "browse_url": ToolContract(
                name="browse_url",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Open a URL and extract page content"
            ),
            "google_search": ToolContract(
                name="google_search",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Search Google and return results"
            ),
            "page_screenshot": ToolContract(
                name="page_screenshot",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Take a screenshot of a web page"
            ),
            "create_pdf": ToolContract(
                name="create_pdf",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.MODERATE,
                description="Create a PDF document from text content"
            ),
            "read_pdf": ToolContract(
                name="read_pdf",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Extract text from a PDF file"
            ),
            "create_image": ToolContract(
                name="create_image",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.MODERATE,
                description="Create an image with optional text"
            ),
            "resize_image": ToolContract(
                name="resize_image",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.MODERATE,
                description="Resize an image to specified dimensions"
            ),
            "capture_screen": ToolContract(
                name="capture_screen",
                preconditions=[],
                postconditions=[lambda obs, b: True],
                risk_level=RiskLevel.SAFE,
                description="Capture a screenshot of the desktop"
            ),
        })
        
        # Initialize executors
        self.file_executor = FileSystemExecutor()
        self.code_interpreter = CodeInterpreter()
        self.system_monitor = SystemMonitor()
        self.clipboard = ClipboardTools()
        self.browser = BrowserTools()
        self.pdf_tools = PDFTools()
        self.image_tools = ImageTools()
        self.screenshot_tools = ScreenshotTools()

    def _get_contract(self, action_name: str) -> ToolContract:
        """
        Get contract for a tool, auto-generating one for unknown tools.
        Uses dynamic risk classification from escalation_guard.
        """
        if action_name in self.contracts:
            return self.contracts[action_name]

        # Dynamic contract generation for unknown tools
        from .escalation_guard import classify_action_risk
        risk = classify_action_risk(action_name)

        if risk <= 2:
            risk_level = RiskLevel.SAFE
        elif risk <= 5:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HAZARDOUS

        contract = ToolContract(
            name=action_name,
            preconditions=[],  # No preconditions for dynamically discovered tools
            postconditions=[lambda obs, b: True],  # Accept any valid output
            risk_level=risk_level,
            description=f"Dynamically classified tool (risk={risk}/10)",
        )
        # Cache for future use
        self.contracts[action_name] = contract
        return contract

    def verify_preconditions(self, action: Action, belief: BeliefState) -> Tuple[bool, str]:
        """Checks if it's safe to execute the action given current beliefs"""
        contract = self._get_contract(action.name)

        for check in contract.preconditions:
            if not check(belief):
                return False, f"Precondition failed for {action.name} under current belief state"

        return True, "Verified"

    def verify_postconditions(self, action: Action, observation: str, belief: BeliefState) -> Tuple[bool, str]:
        """Verifies if the tool execution resulted in an expected state transition"""
        contract = self._get_contract(action.name)

        for check in contract.postconditions:
            if not check(observation, belief):
                return False, f"Postcondition failed for {action.name}: observation '{observation[:30]}...' not valid for contract"

        return True, "Verified"

    def execute(self, action: Action) -> str:
        """Executes the tool and returns the observation string."""
        print(f"  [ToolGate] Executing: {action.name} with {action.arguments}")
        
        try:
            args = action.arguments or {}
            
            # --- File System Operations (via FileSystemExecutor) ---
            if action.name == "list_files":
                result = self.file_executor.list_files(
                    path=args.get("path", "."),
                    pattern=args.get("pattern", "*"),
                    recursive=args.get("recursive", False)
                )
                if result.success and result.data:
                    names = [f['name'] for f in result.data if isinstance(f, dict)]
                    return f"Files found: {names}"
                return str(result)
            
            elif action.name == "read_file":
                filepath = args.get("filepath", "")
                if not filepath:
                    return "Error: No filepath provided"
                result = self.file_executor.read_file(filepath)
                return str(result) if not result.success else f"Contents of {os.path.basename(filepath)}:\n{result.data[:500]}"
            
            elif action.name == "delete_file":
                filepath = args.get("filepath", "")
                if not filepath:
                    return "Error: No filepath provided"
                result = self.file_executor.delete_file(filepath, permanent=False)  # TRASH!
                return str(result)
            
            elif action.name == "move_file":
                src = args.get("src", args.get("source", ""))
                dst = args.get("dst", args.get("destination", ""))
                if not src or not dst:
                    return "Error: Both source and destination required"
                result = self.file_executor.move_file(src, dst)
                return str(result)
            
            elif action.name == "copy_file":
                src = args.get("src", args.get("source", ""))
                dst = args.get("dst", args.get("destination", ""))
                if not src or not dst:
                    return "Error: Both source and destination required"
                result = self.file_executor.copy_file(src, dst)
                return str(result)
            
            elif action.name == "write_file":
                filepath = args.get("filepath", "")
                content = args.get("content", "")
                if not filepath:
                    return "Error: No filepath provided"
                result = self.file_executor.write_file(filepath, content)
                return str(result)
            
            elif action.name == "get_file_info":
                filepath = args.get("filepath", "")
                if not filepath:
                    return "Error: No filepath provided"
                result = self.file_executor.get_file_info(filepath)
                if result.success:
                    info = result.data
                    return f"File Info: {info['name']} | {info['size_human']} | Modified: {info['modified'][:10]} | Type: {'directory' if info['is_dir'] else info.get('extension', 'file')}"
                return str(result)
            
            elif action.name == "search_files":
                path = args.get("path", ".")
                query = args.get("query", "*")
                content_search = args.get("content_search", False)
                result = self.file_executor.search_files(path, query, content_search)
                return str(result)
            
            elif action.name == "create_directory":
                path = args.get("path", "")
                if not path:
                    return "Error: No path provided"
                result = self.file_executor.create_directory(path)
                return str(result)
            
            elif action.name == "get_directory_tree":
                path = args.get("path", ".")
                max_depth = args.get("max_depth", 3)
                result = self.file_executor.get_directory_tree(path, max_depth)
                if result.success:
                    return f"Directory Tree:\n{result.data}"
                return str(result)
            
            # --- Non-file tools ----------------------------
            elif action.name == "web_search":
                if not self.tavily_client:
                    return "Error: Tavily API key not configured."
                query = args.get("query", "")
                response = self.tavily_client.search(query=query, search_depth="basic")
                results = response.get("results", [])[:2]
                return f"Search Results: {results}"
            
            elif action.name == "ask_user":
                question = args.get("question", "")
                return f"CLARIFICATION_REQUEST: {question}"
            
            elif action.name == "answer_user":
                return f"FINAL_ANSWER: {args.get('text', '')}"
            
            # --- Code Execution ----------------------------
            elif action.name == "run_code":
                code = args.get("code", "")
                language = args.get("language", "python")
                if not code:
                    return "Error: No code provided"
                result = self.code_interpreter.execute(code, language=language)
                return str(result)
            
            elif action.name == "run_shell":
                command = args.get("command", "")
                if not command:
                    return "Error: No command provided"
                result = self.code_interpreter.execute(command, language="shell")
                return str(result)
            
            # --- System Tools -----------------------------
            elif action.name == "system_info":
                info = self.system_monitor.get_system_info()
                return str(info)
            
            elif action.name == "get_processes":
                procs = self.system_monitor.get_running_processes(top_n=10)
                if procs:
                    lines = [str(p) for p in procs]
                    return "Top processes:\n" + "\n".join(lines)
                return "No process information available (psutil not installed?)"
            
            elif action.name == "clipboard_read":
                return f"Clipboard: {self.clipboard.get_clipboard()}"
            
            # --- Browser Tools -----------------------------
            elif action.name == "browse_url":
                url = args.get("url", "")
                if not url:
                    return "Error: No URL provided"
                result = self.browser.open_url(url)
                if result.success:
                    return f"Page: {result.data['title']}\nURL: {result.data['url']}\nContent: {result.data['text_preview']}"
                return f"Browse Error: {result.error}"
            
            elif action.name == "google_search":
                query = args.get("query", "")
                if not query:
                    return "Error: No query provided"
                result = self.browser.search_google(query)
                if result.success:
                    data = result.data
                    if data.get("results"):
                        lines = []
                        for r in data["results"]:
                            lines.append(f"{r['position']}. {r['title']}\n   {r['url']}\n   {r['snippet'][:100]}")
                        return f"Google Results for '{query}':\n" + "\n".join(lines)
                    return f"Search completed but no structured results. Raw text: {data.get('raw_text', '')[:500]}"
                return f"Search Error: {result.error}"
            
            elif action.name == "page_screenshot":
                url = args.get("url", "")
                save_path = args.get("save_path", f"screenshot_{url.replace('/', '_')[:20]}.png")
                if not url:
                    return "Error: No URL provided"
                result = self.browser.screenshot_page(url, save_path)
                return result.message if result.success else f"Screenshot Error: {result.error}"
            
            # --- PDF Tools ---------------------------------
            elif action.name == "create_pdf":
                content = args.get("content", "")
                output_path = args.get("output_path", "output.pdf")
                title = args.get("title", "Document")
                if not content:
                    return "Error: No content provided"
                result = self.pdf_tools.create_pdf(content, output_path, title)
                return result.message if result.success else f"PDF Error: {result.error}"
            
            elif action.name == "read_pdf":
                filepath = args.get("filepath", "")
                if not filepath:
                    return "Error: No filepath provided"
                result = self.pdf_tools.read_pdf(filepath)
                if result.success:
                    return f"PDF ({result.data['pages']} pages):\n{result.data['text'][:1000]}"
                return f"PDF Read Error: {result.error}"
            
            # --- Image Tools -------------------------------
            elif action.name == "create_image":
                width = args.get("width", 800)
                height = args.get("height", 600)
                color = args.get("color", "white")
                text = args.get("text", "")
                output_path = args.get("output_path", "output.png")
                result = self.image_tools.create_image(width, height, color, text, output_path)
                return result.message if result.success else f"Image Error: {result.error}"
            
            elif action.name == "resize_image":
                filepath = args.get("filepath", "")
                width = args.get("width", 0)
                height = args.get("height", 0)
                output_path = args.get("output_path", None)
                if not filepath or not width or not height:
                    return "Error: filepath, width, and height required"
                result = self.image_tools.resize_image(filepath, width, height, output_path)
                return result.message if result.success else f"Resize Error: {result.error}"
            
            # --- Screenshot Tools --------------------------
            elif action.name == "capture_screen":
                output_path = args.get("output_path", None)
                result = self.screenshot_tools.capture_screen(output_path)
                return result.message if result.success else f"Screenshot Error: {result.error}"
            
            else:
                return f"Error: Unknown tool '{action.name}'"
                
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"
