"""
System Monitor — Real-time system state awareness.

Gives the agent knowledge about the environment it's running in:
CPU, RAM, disk, processes, environment variables, installed packages.

Inspired by OS-Copilot's system awareness capabilities.
"""

import os
import sys
import platform
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """Snapshot of system state."""
    os_name: str
    os_version: str
    hostname: str
    cpu_count: int
    cpu_percent: float
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    python_version: str
    cwd: str

    def __str__(self):
        return (
            f"System: {self.os_name} {self.os_version} ({self.hostname})\n"
            f"CPU: {self.cpu_count} cores, {self.cpu_percent:.1f}% used\n"
            f"RAM: {self.ram_used_gb:.1f}GB / {self.ram_total_gb:.1f}GB ({self.ram_percent:.0f}%)\n"
            f"Disk: {self.disk_used_gb:.1f}GB / {self.disk_total_gb:.1f}GB ({self.disk_percent:.0f}%)\n"
            f"Python: {self.python_version}\n"
            f"CWD: {self.cwd}"
        )


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    cpu_percent: float
    memory_mb: float
    status: str

    def __str__(self):
        return f"[{self.pid}] {self.name} — CPU: {self.cpu_percent:.1f}%, RAM: {self.memory_mb:.1f}MB ({self.status})"


class SystemMonitor:
    """
    System state awareness for the desktop agent.
    
    Uses psutil for cross-platform system monitoring.
    Falls back to basic os/platform info if psutil unavailable.
    """

    @staticmethod
    def get_system_info() -> SystemInfo:
        """Get comprehensive system snapshot."""
        cpu_count = os.cpu_count() or 1
        cpu_percent = 0.0
        ram_total = ram_used = ram_pct = 0.0
        disk_total = disk_used = disk_pct = 0.0

        if psutil:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            ram_total = mem.total / (1024**3)
            ram_used = mem.used / (1024**3)
            ram_pct = mem.percent

            try:
                disk = psutil.disk_usage(os.path.abspath('.'))
                disk_total = disk.total / (1024**3)
                disk_used = disk.used / (1024**3)
                disk_pct = disk.percent
            except:
                pass

        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            hostname=platform.node(),
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            ram_total_gb=ram_total,
            ram_used_gb=ram_used,
            ram_percent=ram_pct,
            disk_total_gb=disk_total,
            disk_used_gb=disk_used,
            disk_percent=disk_pct,
            python_version=sys.version.split()[0],
            cwd=os.getcwd(),
        )

    @staticmethod
    def get_disk_usage(path: str = ".") -> Dict[str, Any]:
        """Get disk usage for a specific path."""
        if not psutil:
            return {"error": "psutil not installed"}
        try:
            usage = psutil.disk_usage(os.path.abspath(path))
            return {
                "path": os.path.abspath(path),
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent_used": usage.percent,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_running_processes(top_n: int = 15) -> List[ProcessInfo]:
        """Get top N processes by CPU usage."""
        if not psutil:
            return []
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status']):
            try:
                info = proc.info
                mem_mb = (info['memory_info'].rss / (1024**2)) if info.get('memory_info') else 0
                processes.append(ProcessInfo(
                    pid=info['pid'],
                    name=info['name'] or 'unknown',
                    cpu_percent=info.get('cpu_percent', 0) or 0,
                    memory_mb=mem_mb,
                    status=info.get('status', 'unknown'),
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by memory (most reliable), take top N
        processes.sort(key=lambda p: p.memory_mb, reverse=True)
        return processes[:top_n]

    @staticmethod
    def get_environment_variables(filter_prefix: str = None) -> Dict[str, str]:
        """Get environment variables, optionally filtered by prefix."""
        env = dict(os.environ)
        if filter_prefix:
            env = {k: v for k, v in env.items() if k.upper().startswith(filter_prefix.upper())}
        # Mask sensitive values
        sensitive_keys = ['KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'PASS', 'CREDENTIAL']
        for k in env:
            if any(s in k.upper() for s in sensitive_keys):
                env[k] = '***MASKED***'
        return env

    @staticmethod
    def get_installed_packages() -> List[Dict[str, str]]:
        """Get installed Python packages."""
        packages = []
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                packages.append({
                    "name": dist.metadata['Name'],
                    "version": dist.metadata['Version'],
                })
        except:
            # Fallback: use pip
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=columns"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split('\n')[2:]:
                parts = line.split()
                if len(parts) >= 2:
                    packages.append({"name": parts[0], "version": parts[1]})

        return sorted(packages, key=lambda p: p['name'].lower())
