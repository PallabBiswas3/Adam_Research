import ctypes
import importlib.metadata
import platform
import subprocess
from pathlib import Path
import hashlib

import torch


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def process_peak_rss_bytes():
    """Process-lifetime high-water memory, not attributable to one in-process run."""
    if platform.system() == "Windows":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak*1024)


def environment_metadata(device):
    root = Path(__file__).resolve().parents[3]
    # Trust only the user's explicitly selected workspace; no global Git mutation.
    workspace = root.parent
    git = ["git", "-c", f"safe.directory={workspace.as_posix()}"]
    try:
        commit = subprocess.run(git+["rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(git+["status", "--porcelain"], cwd=root,
                                   capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    digest = hashlib.sha256()
    for source in sorted(list((root/"src").rglob("*.py"))+list((root/"scripts").glob("*.py"))):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    versions = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "pandas", "matplotlib", "pytest", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), device=str(device), threads=torch.get_num_threads(),
                cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
                packages=versions, git_commit=commit, git_dirty=dirty, source_sha256=digest.hexdigest())
