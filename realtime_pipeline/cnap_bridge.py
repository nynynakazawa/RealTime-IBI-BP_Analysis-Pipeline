from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Callable


@dataclass(frozen=True)
class CNAPBeatEvent:
    elapsed_time_s: float
    beat_index: int
    systolic: float
    mean: float
    diastolic: float
    heart_rate: float


@dataclass
class CNAPCapture:
    process: subprocess.Popen[str]
    session_id: str
    capture_root: Path
    archive_root: Path
    events: Queue[CNAPBeatEvent]
    _thread: threading.Thread | None = None
    _stop: threading.Event | None = None

    @property
    def beats_csv(self) -> Path:
        return self.archive_root / self.session_id / f"{self.session_id}_beats.csv"

    @property
    def metadata_json(self) -> Path:
        return self.archive_root / self.session_id / f"{self.session_id}_beats.json"

    def drain(self, handler: Callable[[CNAPBeatEvent], None]) -> None:
        while True:
            try:
                handler(self.events.get_nowait())
            except Empty:
                return

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=1)


def _pump_output(process: subprocess.Popen[str], queue: Queue[CNAPBeatEvent], stop: threading.Event) -> None:
    assert process.stdout is not None
    pattern = re.compile(
        r"\[beat\].*t=(?P<elapsed>[0-9.]+)s #(?P<beat>\d+).*"
        r"Sys=(?P<sys>[0-9.]+).*Mean=(?P<mean>[0-9.]+).*Dia=(?P<dia>[0-9.]+).*HR=(?P<hr>[0-9.]+)"
    )
    for line in process.stdout:
        if stop.is_set():
            break
        match = pattern.search(line)
        if not match:
            continue
        queue.put(
            CNAPBeatEvent(
                elapsed_time_s=float(match.group("elapsed")),
                beat_index=int(match.group("beat")),
                systolic=float(match.group("sys")),
                mean=float(match.group("mean")),
                diastolic=float(match.group("dia")),
                heart_rate=float(match.group("hr")),
            )
        )


def start_cnap_capture(repo_root: Path, session_id: str) -> CNAPCapture:
    cnap_dir = repo_root / "CNAP"
    script = repo_root / "CNAP" / "realtime_capture.py"
    venv_python = cnap_dir / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else "python3"
    archive_root = repo_root / "Analysis" / "Data" / "pdp" / "realtime_aux"
    queue: Queue[CNAPBeatEvent] = Queue()
    stop = threading.Event()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [python, "-u", str(script), "--session-id", session_id],
        text=True,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    thread = threading.Thread(target=_pump_output, args=(process, queue, stop), daemon=True)
    thread.start()
    return CNAPCapture(
        process=process,
        session_id=session_id,
        capture_root=repo_root / "CNAP" / "captures",
        archive_root=archive_root,
        events=queue,
        _thread=thread,
        _stop=stop,
    )


def cnap_is_ready(repo_root: Path) -> tuple[bool, str]:
    cnap_dir = repo_root / "CNAP"
    if not cnap_dir.exists():
        return False, f"CNAP directory not found: {cnap_dir}"
    venv_python = cnap_dir / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else "python3"
    check_script = """
from pathlib import Path
import sys

cnap_dir = Path(sys.argv[1])
sys.path.insert(0, str(cnap_dir))

import usb.core
from tusb_adapio import TUSBAdapio, get_default_backend

backend = get_default_backend()
device = usb.core.find(
    idVendor=TUSBAdapio.VID,
    idProduct=TUSBAdapio.PID,
    backend=backend,
)

if device is None:
    print("CNAP AD converter not detected: TUSB-ADAPIO not found", file=sys.stderr)
    raise SystemExit(2)

product = getattr(device, "product", None) or "TUSB-ADAPIO"
serial = getattr(device, "serial_number", None) or "unknown"
bus = getattr(device, "bus", None)
address = getattr(device, "address", None)
print(f"{product} serial={serial} bus={bus} address={address}")
"""
    try:
        result = subprocess.run(
            [python, "-c", check_script, str(cnap_dir)],
            check=False,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )
    except Exception as exc:
        return False, f"failed to run CNAP probe: {exc}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0:
        return True, stdout or "CNAP probe OK"
    if result.returncode == 2:
        return False, stderr or "CNAP AD converter not detected"
    return False, stderr or stdout or "CNAP is not ready"
