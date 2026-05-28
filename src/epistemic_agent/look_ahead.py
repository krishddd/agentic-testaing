import math
import re
import os
from typing import List, Optional, Tuple
import ollama
from .generative_model import BeliefState, Action, ActionType, FileStatus, UserIntent, RiskLevel
from .config import settings
from .uncertainty import UncertaintyEstimator


class LookAheadModule:
    """
    Simulates actions in 'Cognitive Space' to predict outcomes and update Belief States.

    Improvements:
    1. Belief-Driven Action Generation: proposes actions based on entropy of each
       belief factor, not just keyword matching
    2. Entropy-Based Scoring: actions are ranked by their expected entropy reduction
    3. Dynamic Action Space: expands candidates based on belief state analysis
    4. LLM-based state transition simulation (replaces B matrix)
    """

    def __init__(self, uncertainty_estimator: UncertaintyEstimator):
        self.model_name = settings.OLLAMA_MODEL
        self.uncertainty = uncertainty_estimator

    def _calculate_entropy(self, probs: dict) -> float:
        """Shannon entropy H(P) = -S p*ln(p)"""
        h = 0.0
        for p in probs.values():
            if p > 1e-12:
                h -= p * math.log(p)
        return h

    async def predict_outcome_and_update_belief(
        self,
        action: Action,
        current_belief: BeliefState,
        context: str
    ) -> BeliefState:
        """
        Asks LLM: "If I take ACTION in CONTEXT, what is the likely outcome
        and how does it change my beliefs?"

        Returns an updated BeliefState (posterior).
        """
        prompt = f"""
        Analyze the risk and belief shift for the following action in an Active Inference loop.

        Current World Context: {context}

        Current Belief Probabilities:
        - FileStatus: {dict(current_belief.file_status_probs)}
        - UserIntent: {dict(current_belief.user_intent_probs)}
        - RiskLevel: {dict(current_belief.risk_level_probs)}

        Proposed Action: {action.name} ({action.action_type})
        Action Arguments: {action.arguments}

        Task:
        1. Predict if this action will resolve uncertainty or introduce risk.
        2. Provide UPDATED probabilities for the belief factors.

        Output format (STRICT):
        PREDICTION: <Brief prediction of what tool will return>
        UPD_FILE_STATUS: exists: <prob>, does_not_exist: <prob>, ambiguous: <prob>
        UPD_USER_INTENT: delete: <prob>, read: <prob>, clarify: <prob>, unknown: <prob>
        UPD_RISK_LEVEL: safe: <prob>, moderate: <prob>, hazardous: <prob>
        """

        try:
            import asyncio
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a Bayesian mental simulator for an AI agent.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            content = response['message']['content']

            new_belief = current_belief.model_copy(deep=True)

            def parse_probs(line, keys):
                probs = {}
                for key in keys:
                    match = re.search(rf"{key}:\s*([\d\.]+)", line)
                    if match:
                        probs[key] = float(match.group(1))
                total = sum(probs.values())
                if total > 0:
                    return {k: v / total for k, v in probs.items()}
                return None

            for line in content.split('\n'):
                if "UPD_FILE_STATUS" in line:
                    p = parse_probs(line, [e.value for e in FileStatus])
                    if p:
                        new_belief.file_status_probs = {FileStatus(k): v for k, v in p.items()}
                elif "UPD_USER_INTENT" in line:
                    p = parse_probs(line, [e.value for e in UserIntent])
                    if p:
                        new_belief.user_intent_probs = {UserIntent(k): v for k, v in p.items()}
                elif "UPD_RISK_LEVEL" in line:
                    p = parse_probs(line, [e.value for e in RiskLevel])
                    if p:
                        new_belief.risk_level_probs = {RiskLevel(k): v for k, v in p.items()}

            return new_belief

        except Exception as e:
            print(f"Error in LLM Look-Ahead Simulation: {e}")
            return current_belief

    # --- Dynamic Parsing Helpers ----------------------------------

    def _extract_path(self, user_input: str) -> str:
        """
        Extracts target directory path from user input.
        Handles: "from src", "of the src folder", "in my_dir", "at /tmp"
        Only returns paths that look like file-system paths (not English words).
        """
        # Common English words that match preposition patterns but aren't paths
        noise = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'this', 'that', 'those', 'these', 'it', 'its', 'my', 'your',
            'what', 'which', 'who', 'whom', 'how', 'why', 'where', 'when',
            'current', 'here', 'there', 'yesterday', 'today', 'tomorrow',
            'above', 'below', 'underwater', 'general', 'particular',
            'order', 'detail', 'addition', 'fact', 'case', 'terms',
            'thing', 'things', 'something', 'anything', 'everything',
        }

        # Pattern 1: "from/of/in X folder/directory" — X is explicitly a path
        m = re.search(
            r"(?:from|of|in|at|inside)\s+(?:the\s+|my\s+|this\s+)?([A-Za-z0-9_\\\/:\.\-]+?)\s+(?:folder|directory|dir)\b",
            user_input, re.IGNORECASE
        )
        if m:
            path = m.group(1).strip().rstrip('.')
            if path.lower() not in noise:
                return path

        # Pattern 2: "from X" — X must look like a path
        m = re.search(
            r"\b(?:from|inside)\s+(?:the\s+|my\s+)?([A-Za-z0-9_\\\/:\.\-]+)",
            user_input, re.IGNORECASE
        )
        if m:
            path = m.group(1).strip().rstrip('.')
            looks_like_path = any(c in path for c in ['/', '\\', '_', '.']) or len(path) <= 15
            if path.lower() not in noise and looks_like_path:
                return path

        # Pattern 3: "files in X" / "in X directory" — only with file context
        m = re.search(
            r"\b(?:files?|log|csv)\s+in\s+(?:the\s+)?([A-Za-z0-9_\\\/:\.\-]+)",
            user_input, re.IGNORECASE
        )
        if m:
            path = m.group(1).strip().rstrip('.')
            if path.lower() not in noise:
                return path

        return "."

    def _extract_filename(self, user_input: str) -> Optional[str]:
        """
        Extracts target filename from user input.
        Handles many patterns:
          - "delete femp file from test_demo" -> femp
          - "read femp.txt" -> femp.txt
          - "read the contents of app.py" -> app.py
          - "show me config.json" -> config.json
          - "open file named report.pdf" -> report.pdf
          - "cat requirements.txt" -> requirements.txt
        """
        user_input_stripped = user_input.strip()

        # Pattern 1: "contents of X" / "content of X"
        m = re.search(r"contents?\s+of\s+([A-Za-z0-9_\-\.\/\\]+)", user_input_stripped, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Pattern 2: "file named X" / "file called X"
        m = re.search(r"file\s+(?:named|called)\s+([A-Za-z0-9_\-\.\/\\]+)", user_input_stripped, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Pattern 3: Explicit filename with extension anywhere in text (e.g. "app.py", "data.csv")
        m = re.search(r"\b([A-Za-z0-9_\-]+\.[A-Za-z0-9]{1,6})\b", user_input_stripped)
        if m:
            candidate = m.group(1)
            # Exclude common non-filenames
            if candidate.lower() not in {'e.g', 'i.e', 'vs.', 'etc.', 'a.m', 'p.m'}:
                return candidate

        # Pattern 4: Classic verb + [the] [file] + filename [file] [from/in/$]
        m = re.search(
            r"(?:delete|remove|read|open|show|cat|display)\s+(?:the\s+)?(?:file\s+)?([A-Za-z0-9_\-\.]+?)(?:\s+file)?(?:\s+from|\s+in|\s*$)",
            user_input_stripped, re.IGNORECASE
        )
        if m:
            candidate = m.group(1).strip()
            # Filter noise words that aren't filenames
            noise = {'the', 'this', 'that', 'all', 'my', 'some', 'every',
                      'me', 'contents', 'content', 'directory', 'folder', 'tree',
                      'file', 'files'}
            if candidate.lower() not in noise:
                return candidate

        return None

    def _extract_extension(self, user_input: str) -> Optional[str]:
        """
        Extracts file extension filter from user input.
        Only triggers when the query is about listing/finding files of a type.
        "List all Python files" -> "*.py"
        "Show me .csv files" -> "*.csv"
        """
        lower = user_input.lower()

        # Only detect extensions when the query is about files
        file_context = any(w in lower for w in ['file', 'files', 'list', 'find', 'show me all', 'all the'])
        if not file_context:
            return None

        # Skip if we already found a specific filename (e.g. "read settings.yaml")
        # Extension filter is for type-based listing, not individual files
        if self._extract_filename.__func__ != self._extract_extension.__func__:  # avoid recursion
            test_fname = re.search(r"\b[A-Za-z0-9_\-]+\.[A-Za-z0-9]{1,6}\b", user_input)
            if test_fname and any(w in lower for w in ['read', 'open', 'cat', 'show', 'named', 'called']):
                return None

        # Direct extension mention: ".py files", ".txt files"
        m = re.search(r"\.(\w{1,6})\s+file", user_input, re.IGNORECASE)
        if m:
            return f"*.{m.group(1)}"

        # Language-to-extension mapping (word-boundary matching)
        lang_map = [
            (r'\bpython\b', '*.py'),
            (r'\bjavascript\b', '*.js'),
            (r'\btypescript\b', '*.ts'),
            (r'\bjava\b', '*.java'),
            (r'\bc\+\+\b', '*.cpp'),
            (r'\bcpp\b', '*.cpp'),
            (r'\brust\b', '*.rs'),
            (r'\bgolang\b', '*.go'),
            (r'\bruby\b', '*.rb'),
            (r'\btext\b', '*.txt'),
            (r'\bjson\b', '*.json'),
            (r'\byaml\b', '*.yaml'),
            (r'\bcsv\b', '*.csv'),
            (r'\bhtml\b', '*.html'),
            (r'\bcss\b', '*.css'),
            (r'\bmarkdown\b', '*.md'),
            (r'\bxml\b', '*.xml'),
            (r'\bsql\b', '*.sql'),
            (r'\bshell\b', '*.sh'),
            (r'\bbash\b', '*.sh'),
            (r'\blog\b', '*.log'),
            (r'\bpdf\b', '*.pdf'),
        ]

        for pattern, ext in lang_map:
            if re.search(pattern, lower):
                return ext

        return None

    def _is_bulk_reference(self, text: str) -> bool:
        """Detects bulk/wildcard references like 'all files', 'everything', 'every file'."""
        bulk_words = {'all', 'every', 'everything', 'all the', 'entire', 'whole'}
        lower = text.lower().strip()
        return any(w in lower for w in bulk_words)

    # -----------------------------------------------------------------
    #  Dynamic Intent Classification
    #  Keyword families -- each set can be extended independently
    # -----------------------------------------------------------------

    # File operations
    _FILE_OPS = {
        'delete', 'remove', 'read', 'open', 'show', 'list', 'file',
        'folder', 'directory', 'cat', 'display', 'tree', 'copy',
        'move', 'rename', 'create folder', 'mkdir', 'write file',
        'save file', 'edit file', 'view file', 'check file',
        'dir', 'ls', 'find file', 'get file', 'touch',
    }

    # Dangerous shell commands
    _DANGEROUS_CMDS = {
        'rm -rf', 'format c', 'sudo rm', 'del /f', 'rmdir /s',
        'chmod 777', 'mkfs', 'dd if=', ':(){', 'fork bomb',
        'system directory', 'system32', 'windows\\system',
        'deltree', 'rd /s', 'format d', 'diskpart',
        'net user', 'net localgroup', 'reg delete',
        'taskkill /f', 'shutdown', 'poweroff',
    }

    # Code execution
    _CODE_EXEC = {
        'run code', 'run python', 'run script', 'execute code',
        'execute python', 'run shell', 'run command', 'execute command',
        'run javascript', 'run java', 'run bash', 'compile',
        'run program', 'execute script', 'invoke', 'launch script',
        'eval', 'exec', 'run node', 'npm run', 'pip install',
    }

    # System queries
    _SYSTEM_QUERIES = {
        'system info', 'cpu', 'ram', 'memory usage', 'disk usage',
        'process', 'running process', 'clipboard', 'system status',
        'os version', 'kernel', 'uptime', 'disk space', 'free memory',
        'task manager', 'top', 'htop', 'sysinfo', 'hardware',
        'network info', 'ip address', 'hostname', 'environment',
    }

    # Factual / knowledge queries
    _FACTUAL_QUERIES = {
        'what', 'who', 'where', 'when', 'how', 'why', '?',
        'tell me', 'explain', 'population', 'capital', 'president',
        'price', 'cost', 'weather', 'gdp', 'compare', 'define',
        'describe', 'meaning', 'difference between', 'versus',
        'calculate', 'convert', 'translate', 'summarize', 'recap',
        'history of', 'origin of', 'benefits of', 'advantages',
    }

    # Browser / web
    _BROWSER_QUERIES = {
        'browse', 'open url', 'open website', 'visit', 'webpage',
        'google search', 'search google', 'search online', 'look up online',
        'page screenshot', 'screenshot of website', 'capture page',
        'web page', 'website', 'http', 'www.', '.com', '.org', '.net',
        'navigate to', 'go to site', 'open link', 'fetch url',
        'download page', 'scrape', 'crawl', 'load page',
    }

    # PDF operations
    _PDF_QUERIES = {
        'create pdf', 'make pdf', 'generate pdf', 'pdf report',
        'read pdf', 'open pdf', 'extract pdf', 'merge pdf',
        'pdf document', 'convert to pdf', 'pdf file',
        'export pdf', 'print to pdf', 'split pdf', 'combine pdf',
    }

    # Image operations
    _IMAGE_QUERIES = {
        'create image', 'make image', 'generate image', 'new image',
        'resize image', 'crop image', 'convert image', 'watermark',
        'image size', 'photo', 'picture', 'thumbnail',
        'compress image', 'rotate image', 'flip image', 'filter image',
        'image to', 'jpg', 'png', 'gif', 'bmp', 'svg',
    }

    # Screenshot
    _SCREENSHOT_QUERIES = {
        'screenshot', 'screen capture', 'capture screen', 'screengrab',
        'print screen', 'snap screen', 'take a screenshot',
        'screen shot', 'desktop capture', 'window capture',
    }

    def _classify_intent(self, text_lower: str) -> dict:
        """
        Dynamically classify user intent.

        Layer 1: Keyword families (fast ~0ms)
        Layer 2: LLM semantic classification (fallback, ~2s)
                 Only triggers when no keyword matches found.
        """
        def _matches(keyword_set):
            return any(kw in text_lower for kw in keyword_set)

        file_op = _matches(self._FILE_OPS)
        code_exec = _matches(self._CODE_EXEC)
        system_q = _matches(self._SYSTEM_QUERIES)

        result = {
            'file_operation': file_op,
            'dangerous_command': _matches(self._DANGEROUS_CMDS),
            'code_execution': code_exec,
            'system_query': system_q,
            'factual_query': (not file_op and not code_exec and not system_q
                              and _matches(self._FACTUAL_QUERIES)),
            'browser_query': _matches(self._BROWSER_QUERIES),
            'pdf_query': _matches(self._PDF_QUERIES),
            'image_query': _matches(self._IMAGE_QUERIES),
            'screenshot_query': _matches(self._SCREENSHOT_QUERIES),
        }

        # If no intent was matched, use LLM as fallback
        if not any(result.values()):
            result = self._llm_classify_intent(text_lower, result)

        return result

    def _llm_classify_intent(self, text: str, default: dict) -> dict:
        """
        LLM-powered intent classification fallback.
        Only called when keyword matching finds nothing.
        """
        try:
            import json
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You classify user intent for an AI agent. Respond in JSON only."},
                    {"role": "user", "content": f"""Classify this user input into one or more categories:

Input: "{text}"

Categories (set true for matching ones):
- file_operation: reading, writing, deleting, listing, moving files/folders
- dangerous_command: destructive system commands
- code_execution: running code, scripts, or commands
- system_query: CPU, RAM, disk, process information
- factual_query: questions about facts, knowledge, explanations
- browser_query: web browsing, URLs, online search
- pdf_query: creating, reading, merging PDFs
- image_query: creating, editing, resizing images
- screenshot_query: screen capture

Respond JSON:
{{"file_operation": false, "dangerous_command": false, "code_execution": false, "system_query": false, "factual_query": false, "browser_query": false, "pdf_query": false, "image_query": false, "screenshot_query": false}}"""}
                ],
                format="json",
                options={"temperature": 0.0}
            )
            data = json.loads(response["message"]["content"])
            # Merge LLM results (only override false values)
            for key in default:
                if key in data and isinstance(data[key], bool):
                    default[key] = data[key]
            return default
        except Exception:
            # Fallback to factual_query if LLM fails
            default['factual_query'] = True
            return default

    def propose_actions(self, user_input: str, current_belief: BeliefState) -> List[Action]:
        """
        Generates candidate actions based on BOTH input analysis AND belief state entropy.

        Strategy:
        1. Dynamic parsing with multi-pattern extraction
        2. Entropy-driven epistemic action generation
        3. Pragmatic actions from semantic classification
        4. Safety gates for dangerous commands
        5. answer_user gated by evidence + entropy
        """
        actions = []

        # --- Compute entropy for each belief factor ---
        h_file = self._calculate_entropy(dict(current_belief.file_status_probs))
        h_intent = self._calculate_entropy(dict(current_belief.user_intent_probs))
        h_risk = self._calculate_entropy(dict(current_belief.risk_level_probs))

        max_h_file = math.log(len(FileStatus))
        max_h_intent = math.log(len(UserIntent))
        max_h_risk = math.log(len(RiskLevel))

        u_file = h_file / max_h_file if max_h_file > 0 else 0
        u_intent = h_intent / max_h_intent if max_h_intent > 0 else 0
        u_risk = h_risk / max_h_risk if max_h_risk > 0 else 0

        # --- Dynamic extraction ---
        suggested_path = self._extract_path(user_input)
        target_filename = self._extract_filename(user_input)
        extension_filter = self._extract_extension(user_input)

        # --- Dynamic Intent Classification ---
        user_input_lower = user_input.lower()
        intents = self._classify_intent(user_input_lower)
        is_file_operation = intents.get('file_operation', False)
        is_dangerous_command = intents.get('dangerous_command', False)
        is_code_execution = intents.get('code_execution', False)
        is_system_query = intents.get('system_query', False)
        is_factual_query = intents.get('factual_query', False)
        is_browser_query = intents.get('browser_query', False)
        is_pdf_query = intents.get('pdf_query', False)
        is_image_query = intents.get('image_query', False)
        is_screenshot_query = intents.get('screenshot_query', False)

        # --- SAFETY GATE: Block dangerous commands early ---
        if is_dangerous_command:
            actions.append(Action(
                name="ask_user",
                action_type=ActionType.EPISTEMIC,
                arguments={"question": f"[!] This request appears potentially destructive: '{user_input[:60]}'. Are you absolutely sure you want to proceed?"},
                description="Safety gate: dangerous command detected"
            ))
            # Still allow answer_user to explain why it's dangerous
            actions.append(Action(
                name="answer_user",
                action_type=ActionType.PRAGMATIC,
                arguments={"text": "This request involves a potentially dangerous operation."},
                description="Explain safety concerns"
            ))
            # Store for metrics
            self.last_policy_scores = {"current_h": h_file + h_intent + h_risk, "actions": []}
            return actions

        # --- EPISTEMIC ACTIONS -----------------------------

        # list_files: targeted or generic
        if is_file_operation and target_filename:
            actions.append(Action(
                name="list_files",
                action_type=ActionType.EPISTEMIC,
                arguments={"path": suggested_path, "pattern": f"{target_filename}*"},
                description=f"Search for '{target_filename}*' in {suggested_path}"
            ))
        elif is_file_operation and extension_filter:
            actions.append(Action(
                name="list_files",
                action_type=ActionType.EPISTEMIC,
                arguments={"path": suggested_path, "pattern": extension_filter},
                description=f"List {extension_filter} files in {suggested_path}"
            ))
        elif is_file_operation:
            actions.append(Action(
                name="list_files",
                action_type=ActionType.EPISTEMIC,
                arguments={"path": suggested_path},
                description=f"Check files in {suggested_path} (file uncertainty: {u_file:.0%})"
            ))

        # web_search: for factual/knowledge queries
        if not is_file_operation and not is_code_execution and not is_system_query:
            actions.append(Action(
                name="web_search",
                action_type=ActionType.EPISTEMIC,
                arguments={"query": user_input},
                description=f"Search for information (risk uncertainty: {u_risk:.0%})"
            ))
            if is_factual_query:
                actions.append(Action(
                    name="web_search",
                    action_type=ActionType.EPISTEMIC,
                    arguments={"query": f"{user_input} facts statistics"},
                    description="Refined factual search"
                ))

        # ask_user: only when genuinely unclear
        ambiguous_prob = current_belief.file_status_probs.get(FileStatus.AMBIGUOUS, 0.0)
        exists_prob = current_belief.file_status_probs.get(FileStatus.EXISTS, 0.0)
        not_exists_prob = current_belief.file_status_probs.get(FileStatus.DOES_NOT_EXIST, 0.0)
        beliefs_are_clear = (exists_prob > 0.6) or (not_exists_prob > 0.6)

        if is_file_operation and ambiguous_prob > 0.4:
            actions.append(Action(
                name="ask_user",
                action_type=ActionType.EPISTEMIC,
                arguments={"question": "Multiple files match your request. Which file exactly would you like to operate on?"},
                description=f"Targeted file clarification (ambiguous={ambiguous_prob:.0%})"
            ))
        elif not beliefs_are_clear and u_intent > 0.9:
            actions.append(Action(
                name="ask_user",
                action_type=ActionType.EPISTEMIC,
                arguments={"question": f"I want to make sure I understand: what specifically would you like me to do regarding '{user_input[:50]}'?"},
                description="Detailed intent clarification"
            ))

        # Belief-driven risk search
        if u_risk > 0.7 and not is_file_operation:
            actions.append(Action(
                name="web_search",
                action_type=ActionType.EPISTEMIC,
                arguments={"query": f"safety risks of {user_input}"},
                description="Targeted risk assessment search"
            ))

        # --- PRAGMATIC ACTIONS -----------------------------

        # delete_file
        if 'delete' in user_input_lower or 'remove' in user_input_lower:
            delete_target = target_filename
            if not delete_target:
                # Fallback: try the broader delete regex
                dm = re.search(
                    r"(?:delete|remove)\s+(?:the\s+)?(?:file\s+)?([A-Za-z0-9_\\\\/:\.\-]+?)(?:\s+file)?(?:\s+from|\s+in|\s*$)",
                    user_input, re.IGNORECASE
                )
                delete_target = dm.group(1).strip() if dm else None

            if delete_target and not self._is_bulk_reference(delete_target):
                full_path = delete_target
                if suggested_path != "." and not os.path.isabs(delete_target):
                    full_path = os.path.join(suggested_path, delete_target)
                actions.append(Action(
                    name="delete_file",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"filepath": full_path},
                    description=f"Execute deletion of {full_path}"
                ))
            elif delete_target and self._is_bulk_reference(delete_target):
                # Bulk delete detected -> force clarification instead
                actions.append(Action(
                    name="ask_user",
                    action_type=ActionType.EPISTEMIC,
                    arguments={"question": "Bulk deletion detected. Please specify exactly which files you want to delete."},
                    description="Safety: bulk delete requires clarification"
                ))

        # read_file
        if any(w in user_input_lower for w in ['read', 'open', 'show', 'display', 'cat']):
            read_target = target_filename
            if read_target:
                read_path = read_target
                if suggested_path != "." and not os.path.isabs(read_target):
                    read_path = os.path.join(suggested_path, read_target)
                actions.append(Action(
                    name="read_file",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"filepath": read_path},
                    description=f"Read contents of {read_path}"
                ))

        # move_file / rename
        if any(w in user_input_lower for w in ['move', 'rename', 'mv']):
            src_target = target_filename or "unknown_file"
            dst_match = re.search(r"(?:to|into)\s+([A-Za-z0-9_\\\\/:\.\-]+)", user_input, re.IGNORECASE)
            dst = dst_match.group(1).strip() if dst_match else "."
            actions.append(Action(
                name="move_file",
                action_type=ActionType.PRAGMATIC,
                arguments={"src": src_target, "dst": dst},
                description=f"Move {src_target} -> {dst}"
            ))

        # copy_file
        if any(w in user_input_lower for w in ['copy', 'duplicate', 'cp', 'backup']):
            src_target = target_filename or "unknown_file"
            dst_match = re.search(r"(?:to|into|as)\s+([A-Za-z0-9_\\\\/:\.\-]+)", user_input, re.IGNORECASE)
            dst = dst_match.group(1).strip() if dst_match else f"{src_target}.copy"
            actions.append(Action(
                name="copy_file",
                action_type=ActionType.PRAGMATIC,
                arguments={"src": src_target, "dst": dst},
                description=f"Copy {src_target} -> {dst}"
            ))

        # create_directory
        if any(w in user_input_lower for w in ['create folder', 'make folder', 'create directory', 'make directory', 'mkdir', 'new folder']):
            dir_match = re.search(r"(?:called|named|folder|directory)\s+([A-Za-z0-9_\\\\/:\.\-]+?)(?:\s|$)", user_input, re.IGNORECASE)
            dir_name = dir_match.group(1).strip() if dir_match else "new_folder"
            dir_path = dir_name
            if suggested_path != "." and not os.path.isabs(dir_name):
                dir_path = os.path.join(suggested_path, dir_name)
            actions.append(Action(
                name="create_directory",
                action_type=ActionType.PRAGMATIC,
                arguments={"path": dir_path},
                description=f"Create directory: {dir_path}"
            ))

        # get_file_info
        if any(w in user_input_lower for w in ['size', 'info', 'details', 'when was', 'how big', 'how large']):
            info_target = target_filename or "."
            if suggested_path != "." and not os.path.isabs(info_target):
                info_target = os.path.join(suggested_path, info_target)
            actions.append(Action(
                name="get_file_info",
                action_type=ActionType.PRAGMATIC,
                arguments={"filepath": info_target},
                description=f"Get info for {info_target}"
            ))

        # get_directory_tree
        if any(w in user_input_lower for w in ['tree', 'structure', 'layout', 'hierarchy', 'what files', 'what is in']):
            actions.append(Action(
                name="get_directory_tree",
                action_type=ActionType.PRAGMATIC,
                arguments={"path": suggested_path, "max_depth": 3},
                description=f"Show directory structure of {suggested_path}"
            ))

        # search_files
        if any(w in user_input_lower for w in ['find', 'search for', 'grep', 'search in', 'look for', 'containing']):
            search_match = re.search(r"(?:find|search|grep|look for|containing)\s+(?:for\s+)?['\"]?([A-Za-z0-9_\s\.]+?)['\"]?\s*(?:in|$)", user_input, re.IGNORECASE)
            query = search_match.group(1).strip() if search_match else user_input[:30]
            content_search = any(w in user_input_lower for w in ['containing', 'inside', 'content', 'in files', 'grep'])
            actions.append(Action(
                name="search_files",
                action_type=ActionType.PRAGMATIC,
                arguments={"path": suggested_path, "query": query, "content_search": content_search},
                description=f"Search for '{query}' in {suggested_path}"
            ))

        # run_code / run_shell
        if is_code_execution:
            code_match = re.search(r"(?:run|execute)\s+(?:python\s+|shell\s+|bash\s+)?(?:code|script|command)?\s*[:\-]?\s*(.*?)$", user_input, re.IGNORECASE)
            code_hint = code_match.group(1).strip() if code_match else ""
            if 'shell' in user_input_lower or 'bash' in user_input_lower or 'command' in user_input_lower:
                actions.append(Action(
                    name="run_shell",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"command": code_hint or "echo 'specify command'"},
                    description=f"Run shell command"
                ))
            else:
                actions.append(Action(
                    name="run_code",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"code": code_hint or "print('specify code')", "language": "python"},
                    description=f"Run Python code"
                ))

        # system tools
        if is_system_query:
            if any(w in user_input_lower for w in ['process', 'running process']):
                actions.append(Action(
                    name="get_processes",
                    action_type=ActionType.PRAGMATIC,
                    arguments={},
                    description="List running processes"
                ))
            elif 'clipboard' in user_input_lower:
                actions.append(Action(
                    name="clipboard_read",
                    action_type=ActionType.PRAGMATIC,
                    arguments={},
                    description="Read clipboard contents"
                ))
            else:
                actions.append(Action(
                    name="system_info",
                    action_type=ActionType.PRAGMATIC,
                    arguments={},
                    description="Get system information"
                ))

        # --- Browser Tools -----------------------------
        if is_browser_query:
            # Extract URL if present
            url_match = re.search(r'(https?://\S+|www\.\S+|\S+\.com\S*|\S+\.org\S*|\S+\.net\S*)', user_input, re.IGNORECASE)
            extracted_url = url_match.group(1) if url_match else ""

            if 'google search' in user_input_lower or 'search google' in user_input_lower or 'search online' in user_input_lower:
                search_q = re.sub(r'(google search|search google|search online|for)\s*', '', user_input, flags=re.IGNORECASE).strip()
                actions.append(Action(
                    name="google_search",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"query": search_q or user_input},
                    description=f"Search Google for: {search_q[:50]}"
                ))
            elif 'screenshot' in user_input_lower and extracted_url:
                actions.append(Action(
                    name="page_screenshot",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"url": extracted_url, "save_path": "page_screenshot.png"},
                    description=f"Screenshot of {extracted_url}"
                ))
            elif extracted_url:
                actions.append(Action(
                    name="browse_url",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"url": extracted_url},
                    description=f"Browse {extracted_url}"
                ))
            else:
                actions.append(Action(
                    name="browse_url",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"url": user_input.strip()},
                    description="Browse URL from input"
                ))

        # --- PDF Tools ---------------------------------
        if is_pdf_query:
            if any(w in user_input_lower for w in ['create', 'make', 'generate', 'convert to']):
                content_match = re.search(r'(?:content|text|with)\s*[:\-]?\s*(.+?)(?:$|save|output)', user_input, re.IGNORECASE)
                content = content_match.group(1).strip() if content_match else f"Report generated for: {user_input}"
                title_match = re.search(r'(?:title|titled|called)\s*[:\-]?\s*(.+?)(?:$|\s+with)', user_input, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "Generated Report"
                actions.append(Action(
                    name="create_pdf",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"content": content, "output_path": "output.pdf", "title": title},
                    description=f"Create PDF: {title}"
                ))
            elif any(w in user_input_lower for w in ['read', 'open', 'extract']):
                pdf_file = target_filename or "document.pdf"
                if suggested_path != "." and not os.path.isabs(pdf_file):
                    pdf_file = os.path.join(suggested_path, pdf_file)
                actions.append(Action(
                    name="read_pdf",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"filepath": pdf_file},
                    description=f"Read PDF: {pdf_file}"
                ))

        # --- Image Tools -------------------------------
        if is_image_query:
            if any(w in user_input_lower for w in ['create', 'make', 'generate', 'new']):
                text_match = re.search(r'(?:text|saying|with text)\s*[:\-]?\s*["\']?(.+?)["\']?(?:$|\s+on)', user_input, re.IGNORECASE)
                img_text = text_match.group(1).strip() if text_match else ""
                actions.append(Action(
                    name="create_image",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"width": 800, "height": 600, "color": "white", "text": img_text, "output_path": "output.png"},
                    description=f"Create image{' with text: ' + img_text[:30] if img_text else ''}"
                ))
            elif 'resize' in user_input_lower:
                size_match = re.search(r'(\d+)\s*[xX×]\s*(\d+)', user_input)
                w = int(size_match.group(1)) if size_match else 400
                h = int(size_match.group(2)) if size_match else 400
                img_file = target_filename or "image.png"
                if suggested_path != "." and not os.path.isabs(img_file):
                    img_file = os.path.join(suggested_path, img_file)
                actions.append(Action(
                    name="resize_image",
                    action_type=ActionType.PRAGMATIC,
                    arguments={"filepath": img_file, "width": w, "height": h},
                    description=f"Resize {img_file} to {w}x{h}"
                ))

        # --- Screenshot Tools --------------------------
        if is_screenshot_query and not is_browser_query:
            actions.append(Action(
                name="capture_screen",
                action_type=ActionType.PRAGMATIC,
                arguments={"output_path": "desktop_screenshot.png"},
                description="Capture desktop screenshot"
            ))

        # answer_user: available when we have enough context
        total_uncertainty = (u_file + u_intent + u_risk) / 3.0
        is_question = any(w in user_input_lower for w in ['what', 'who', 'where', 'when', 'how', 'why', '?', 'tell me', 'explain'])
        if total_uncertainty < 0.85 or is_question:
            actions.append(Action(
                name="answer_user",
                action_type=ActionType.PRAGMATIC,
                arguments={"text": "Here is your answer based on gathered information..."},
                description=f"Provide answer (avg uncertainty: {total_uncertainty:.0%})"
            ))

        # -- Depth-1 Policy Tree: Simulated Entropy Scoring --
        from .free_energy import FreeEnergyCalculator
        _sim = FreeEnergyCalculator()

        scored = []
        current_h = h_file + h_intent + h_risk
        for a in actions:
            predicted = _sim._simulate_posterior(a, current_belief)
            pred_h = (
                self._calculate_entropy(predicted["file_status"])
                + self._calculate_entropy(predicted["user_intent"])
                + self._calculate_entropy(predicted["risk_level"])
            )
            delta_h = current_h - pred_h
            scored.append((a.name, delta_h, pred_h))

        # Deduplicate by action name (keep best DH)
        seen = {}
        for name, dh, ph in scored:
            if name not in seen or dh > seen[name][0]:
                seen[name] = (dh, ph)
        deduped = [(n, dh, ph) for n, (dh, ph) in seen.items()]
        deduped.sort(key=lambda x: -x[1])

        self.last_policy_scores = {
            "current_h": current_h,
            "actions": deduped
        }

        return actions
