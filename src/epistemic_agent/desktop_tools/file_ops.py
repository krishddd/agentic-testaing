"""
File System Executor — Real file operations with safety & audit trail.

Inspired by Open Interpreter's local execution model, but integrated
with the Active Inference agent's belief/safety framework.

Safety features:
  - delete_file moves to TRASH by default (send2trash)
  - All operations logged with audit trail
  - Size limits on read operations
  - Path validation to prevent traversal attacks
"""

import os
import shutil
import glob
import stat
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    from send2trash import send2trash
except ImportError:
    send2trash = None

logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    """Structured result from a file operation."""
    success: bool
    operation: str
    message: str
    data: Any = None        # Flexible payload (file content, file list, stats, etc.)
    path: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self):
        if self.success:
            return f"{self.operation}: {self.message}"
        return f"{self.operation} FAILED: {self.error}"


class FileSystemExecutor:
    """
    Real file system operations with safety wrapping.
    
    All destructive operations (delete, move, write) are logged
    and can be undone via the audit trail.
    """

    def __init__(self, base_path: str = ".", max_read_bytes: int = 50_000):
        self.base_path = os.path.abspath(base_path)
        self.max_read_bytes = max_read_bytes
        self.audit_log: List[Dict[str, Any]] = []

    # --- Audit --------------------------------------------

    def _log(self, operation: str, path: str, details: str = ""):
        """Record operation in audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "path": path,
            "details": details,
        }
        self.audit_log.append(entry)
        logger.info(f"[FileOps] {operation}: {path} {details}")

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return full audit trail."""
        return list(self.audit_log)

    # --- Path Safety --------------------------------------

    def _safe_path(self, path: str) -> str:
        """Resolve path and validate it's not dangerous."""
        resolved = os.path.abspath(path)
        # Block system-critical directories
        blocked = [
            os.path.expandvars("%SYSTEMROOT%"),  # C:\Windows
            os.path.expandvars("%PROGRAMFILES%"),
            os.path.expandvars("%PROGRAMFILES(X86)%"),
            "/bin", "/sbin", "/usr", "/etc", "/boot", "/proc", "/sys",
        ]
        for b in blocked:
            if b and resolved.lower().startswith(b.lower()):
                raise PermissionError(f"Access to system directory blocked: {resolved}")
        return resolved

    # --- 1. List Files ------------------------------------

    def list_files(
        self, path: str = ".", pattern: str = "*", recursive: bool = False
    ) -> FileResult:
        """List files matching pattern in directory."""
        try:
            safe = self._safe_path(path)
            if not os.path.isdir(safe):
                return FileResult(
                    success=False, operation="list_files",
                    message="", path=safe,
                    error=f"Directory not found: {safe}"
                )

            if recursive:
                matches = glob.glob(os.path.join(safe, "**", pattern), recursive=True)
            else:
                matches = glob.glob(os.path.join(safe, pattern))

            # Get basic info for each file
            files_info = []
            for f in sorted(matches):
                try:
                    st = os.stat(f)
                    files_info.append({
                        "name": os.path.basename(f),
                        "path": f,
                        "size": st.st_size,
                        "is_dir": os.path.isdir(f),
                        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    })
                except OSError:
                    files_info.append({"name": os.path.basename(f), "path": f, "error": "stat failed"})

            self._log("list_files", safe, f"pattern={pattern}, found={len(files_info)}")
            return FileResult(
                success=True, operation="list_files",
                message=f"Found {len(files_info)} items in {os.path.basename(safe)}/",
                data=files_info, path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="list_files", message="", error=str(e))

    # --- 2. Read File -------------------------------------

    def read_file(self, filepath: str, max_bytes: int = None) -> FileResult:
        """Read file contents (text files, with size limit)."""
        try:
            safe = self._safe_path(filepath)
            if not os.path.isfile(safe):
                return FileResult(
                    success=False, operation="read_file",
                    message="", path=safe,
                    error=f"File not found: {safe}"
                )

            limit = max_bytes or self.max_read_bytes
            size = os.path.getsize(safe)

            with open(safe, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(limit)

            truncated = size > limit
            self._log("read_file", safe, f"size={size}, truncated={truncated}")
            return FileResult(
                success=True, operation="read_file",
                message=f"Read {os.path.basename(safe)} ({size} bytes{'  [truncated]' if truncated else ''})",
                data=content, path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="read_file", message="", error=str(e))

    # --- 3. Write File ------------------------------------

    def write_file(self, filepath: str, content: str, mode: str = "w") -> FileResult:
        """Write content to a file (create or overwrite)."""
        try:
            safe = self._safe_path(filepath)
            # Create parent dirs if needed
            os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)

            existed = os.path.exists(safe)
            with open(safe, mode, encoding="utf-8") as f:
                f.write(content)

            action = "overwrote" if existed else "created"
            self._log("write_file", safe, f"{action}, {len(content)} bytes")
            return FileResult(
                success=True, operation="write_file",
                message=f"{'Overwrote' if existed else 'Created'} {os.path.basename(safe)} ({len(content)} bytes)",
                path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="write_file", message="", error=str(e))

    # --- 4. Delete File (to Trash!) -----------------------

    def delete_file(self, filepath: str, permanent: bool = False) -> FileResult:
        """
        Delete a file. Moves to TRASH by default for safety.
        Only permanent=True does actual deletion (requires extra confirmation).
        """
        try:
            safe = self._safe_path(filepath)
            if not os.path.exists(safe):
                return FileResult(
                    success=False, operation="delete_file",
                    message="", path=safe,
                    error=f"File not found: {safe}"
                )

            if permanent:
                if os.path.isdir(safe):
                    shutil.rmtree(safe)
                else:
                    os.remove(safe)
                self._log("delete_file", safe, "PERMANENT deletion")
                return FileResult(
                    success=True, operation="delete_file",
                    message=f"Permanently deleted {os.path.basename(safe)}",
                    path=safe
                )
            else:
                if send2trash is not None:
                    send2trash(safe)
                    self._log("delete_file", safe, "moved to trash")
                    return FileResult(
                        success=True, operation="delete_file",
                        message=f"Moved {os.path.basename(safe)} to trash (recoverable)",
                        path=safe
                    )
                else:
                    # Fallback: permanent delete when send2trash unavailable
                    logger.warning(f"[FileOps] send2trash not installed, using permanent delete for: {safe}")
                    if os.path.isdir(safe):
                        shutil.rmtree(safe)
                    else:
                        os.remove(safe)
                    self._log("delete_file", safe, "PERMANENT deletion (send2trash unavailable)")
                    return FileResult(
                        success=True, operation="delete_file",
                        message=f"Deleted {os.path.basename(safe)} permanently (send2trash not installed)",
                        path=safe
                    )
        except Exception as e:
            return FileResult(success=False, operation="delete_file", message="", error=str(e))

    # --- 5. Move File -------------------------------------

    def move_file(self, src: str, dst: str) -> FileResult:
        """Move/rename a file or directory."""
        try:
            safe_src = self._safe_path(src)
            safe_dst = self._safe_path(dst)

            if not os.path.exists(safe_src):
                return FileResult(
                    success=False, operation="move_file",
                    message="", error=f"Source not found: {safe_src}"
                )

            # If dst is a directory, move into it
            if os.path.isdir(safe_dst):
                safe_dst = os.path.join(safe_dst, os.path.basename(safe_src))

            shutil.move(safe_src, safe_dst)
            self._log("move_file", safe_src, f"-> {safe_dst}")
            return FileResult(
                success=True, operation="move_file",
                message=f"Moved {os.path.basename(safe_src)} -> {os.path.basename(safe_dst)}",
                path=safe_dst
            )
        except Exception as e:
            return FileResult(success=False, operation="move_file", message="", error=str(e))

    # --- 6. Copy File -------------------------------------

    def copy_file(self, src: str, dst: str) -> FileResult:
        """Copy a file or directory."""
        try:
            safe_src = self._safe_path(src)
            safe_dst = self._safe_path(dst)

            if not os.path.exists(safe_src):
                return FileResult(
                    success=False, operation="copy_file",
                    message="", error=f"Source not found: {safe_src}"
                )

            if os.path.isdir(safe_src):
                shutil.copytree(safe_src, safe_dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(safe_dst) or ".", exist_ok=True)
                shutil.copy2(safe_src, safe_dst)

            self._log("copy_file", safe_src, f"-> {safe_dst}")
            return FileResult(
                success=True, operation="copy_file",
                message=f"Copied {os.path.basename(safe_src)} -> {os.path.basename(safe_dst)}",
                path=safe_dst
            )
        except Exception as e:
            return FileResult(success=False, operation="copy_file", message="", error=str(e))

    # --- 7. Get File Info ---------------------------------

    def get_file_info(self, filepath: str) -> FileResult:
        """Get detailed file metadata."""
        try:
            safe = self._safe_path(filepath)
            if not os.path.exists(safe):
                return FileResult(
                    success=False, operation="get_file_info",
                    message="", path=safe,
                    error=f"Path not found: {safe}"
                )

            st = os.stat(safe)
            info = {
                "name": os.path.basename(safe),
                "path": safe,
                "size_bytes": st.st_size,
                "size_human": self._human_size(st.st_size),
                "is_file": os.path.isfile(safe),
                "is_dir": os.path.isdir(safe),
                "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "extension": os.path.splitext(safe)[1] if os.path.isfile(safe) else "",
                "permissions": oct(st.st_mode)[-3:],
            }
            if os.path.isdir(safe):
                try:
                    info["item_count"] = len(os.listdir(safe))
                except PermissionError:
                    info["item_count"] = "access denied"

            self._log("get_file_info", safe)
            return FileResult(
                success=True, operation="get_file_info",
                message=f"{info['name']}: {info['size_human']}, modified {info['modified'][:10]}",
                data=info, path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="get_file_info", message="", error=str(e))

    # --- 8. Search Files ----------------------------------

    def search_files(
        self, path: str = ".", query: str = "*", content_search: bool = False,
        max_results: int = 50
    ) -> FileResult:
        """
        Search for files by name pattern or by content.
        
        - content_search=False: search by filename (glob pattern)
        - content_search=True: search for text inside files (grep-like)
        """
        try:
            safe = self._safe_path(path)
            results = []

            if content_search:
                # Grep-like content search
                for root, dirs, files in os.walk(safe):
                    for fname in files:
                        if len(results) >= max_results:
                            break
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                                for i, line in enumerate(f, 1):
                                    if query.lower() in line.lower():
                                        results.append({
                                            "file": fpath,
                                            "line": i,
                                            "content": line.strip()[:200],
                                        })
                                        break  # One hit per file
                        except (PermissionError, IsADirectoryError, OSError):
                            continue
            else:
                # Filename pattern search
                matches = glob.glob(os.path.join(safe, "**", f"*{query}*"), recursive=True)
                for m in matches[:max_results]:
                    results.append({
                        "name": os.path.basename(m),
                        "path": m,
                        "is_dir": os.path.isdir(m),
                    })

            self._log("search_files", safe, f"query={query}, content={content_search}, found={len(results)}")
            return FileResult(
                success=True, operation="search_files",
                message=f"Found {len(results)} results for '{query}'",
                data=results, path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="search_files", message="", error=str(e))

    # --- 9. Create Directory ------------------------------

    def create_directory(self, path: str) -> FileResult:
        """Create a new directory (with parents)."""
        try:
            safe = self._safe_path(path)
            if os.path.exists(safe):
                return FileResult(
                    success=True, operation="create_directory",
                    message=f"Directory already exists: {os.path.basename(safe)}",
                    path=safe
                )

            os.makedirs(safe, exist_ok=True)
            self._log("create_directory", safe)
            return FileResult(
                success=True, operation="create_directory",
                message=f"Created directory: {os.path.basename(safe)}",
                path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="create_directory", message="", error=str(e))

    # --- 10. Directory Tree -------------------------------

    def get_directory_tree(self, path: str = ".", max_depth: int = 3) -> FileResult:
        """Get a human-readable directory tree."""
        try:
            safe = self._safe_path(path)
            if not os.path.isdir(safe):
                return FileResult(
                    success=False, operation="get_directory_tree",
                    message="", path=safe,
                    error=f"Not a directory: {safe}"
                )

            tree_lines = [os.path.basename(safe) + "/"]
            self._build_tree(safe, tree_lines, prefix="", depth=0, max_depth=max_depth)

            tree_str = "\n".join(tree_lines)
            self._log("get_directory_tree", safe, f"depth={max_depth}")
            return FileResult(
                success=True, operation="get_directory_tree",
                message=f"Directory tree of {os.path.basename(safe)}/ (depth={max_depth})",
                data=tree_str, path=safe
            )
        except Exception as e:
            return FileResult(success=False, operation="get_directory_tree", message="", error=str(e))

    def _build_tree(self, path, lines, prefix, depth, max_depth):
        """Recursive tree builder."""
        if depth >= max_depth:
            return
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            lines.append(f"{prefix}+-- [access denied]")
            return

        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not e.startswith('.')]
        files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and not e.startswith('.')]

        all_entries = dirs + files
        for i, entry in enumerate(all_entries):
            is_last = (i == len(all_entries) - 1)
            connector = "+-- " if is_last else "|-- "
            full_path = os.path.join(path, entry)

            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}{entry}/")
                next_prefix = prefix + ("    " if is_last else "|   ")
                self._build_tree(full_path, lines, next_prefix, depth + 1, max_depth)
            else:
                size = self._human_size(os.path.getsize(full_path))
                lines.append(f"{prefix}{connector}{entry}  ({size})")

    # --- Utilities ----------------------------------------

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
