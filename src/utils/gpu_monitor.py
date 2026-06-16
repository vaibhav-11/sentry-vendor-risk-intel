"""
Background GPU VRAM monitor (AMD ROCm).

Polls `rocm-smi` on a daemon thread every few seconds, summing used/total VRAM
across all GPU indices, and tracks the peak used figure for the run. Each poll is
appended to logs/gpu_memory.log.

This is GPU-side instrumentation only — it has no effect when rocm-smi is absent
(the mock / CPU-only path), where it disables itself silently on the first poll
and every public accessor returns 0. The thread is a daemon so it can never block
process exit.
"""

import logging
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root: src/utils/gpu_monitor.py → parents[2]
_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "gpu_memory.log"

_POLL_INTERVAL_SECONDS = 2.0

# rocm-smi --showmeminfo vram --noheader emits lines like:
#   GPU[0]          : VRAM Total Memory (VRAM): 196608 MB
#   GPU[0]          : VRAM Total Used Memory (VRAM): 1234 MB
_USED_RE  = re.compile(r"GPU\[(\d+)\].*Used Memory \(VRAM\):\s*(\d+)\s*MB", re.IGNORECASE)
_TOTAL_RE = re.compile(r"GPU\[(\d+)\].*Total Memory \(VRAM\):\s*(\d+)\s*MB", re.IGNORECASE)


class GPUMonitor:
    """Polls rocm-smi on a background daemon thread, tracking peak VRAM usage."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._available = True
        self._peak_mb = 0
        self._total_mb = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background daemon thread and begin polling."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="gpu-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to exit and block until it joins."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join()
            self._thread = None

    def peak_mb(self) -> int:
        """Peak VRAM used in MB across the run, or 0 if rocm-smi is unavailable."""
        with self._lock:
            return self._peak_mb if self._available else 0

    def total_mb(self) -> int:
        """Total VRAM in MB, or 0 if rocm-smi is unavailable."""
        with self._lock:
            return self._total_mb if self._available else 0

    # ── Internals ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        # Poll immediately, then every _POLL_INTERVAL_SECONDS until stopped. The
        # Event.wait return value lets the interval double as the exit signal.
        while True:
            self._poll_once()
            if not self._available:
                return
            if self._stop_event.wait(_POLL_INTERVAL_SECONDS):
                return

    def _poll_once(self) -> None:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--noheader"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            # rocm-smi not on this machine — disable silently and stop polling.
            with self._lock:
                self._available = False
            return
        except Exception as e:  # noqa: BLE001 — never let monitoring crash the run
            logger.debug(f"GPU poll failed: {e}")
            return

        used_by_gpu: dict[str, int] = {}
        total_by_gpu: dict[str, int] = {}
        for line in result.stdout.splitlines():
            m_used = _USED_RE.search(line)
            if m_used:
                used_by_gpu[m_used.group(1)] = int(m_used.group(2))
                continue
            m_total = _TOTAL_RE.search(line)
            if m_total:
                total_by_gpu[m_total.group(1)] = int(m_total.group(2))

        used = sum(used_by_gpu.values())
        total = sum(total_by_gpu.values())

        with self._lock:
            if used > self._peak_mb:
                self._peak_mb = used
            if total > 0:
                self._total_mb = total

        self._write_log_line(used, total)

    def _write_log_line(self, used: int, total: int) -> None:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H:%M:%S")
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{ts} used={used}MB total={total}MB\n")
        except OSError as e:
            logger.debug(f"Could not write GPU log line: {e}")
