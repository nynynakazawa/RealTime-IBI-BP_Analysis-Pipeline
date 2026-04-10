from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
import os
import re
import signal
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable


@dataclass(frozen=True)
class CNAPBeatEvent:
    timestamp_ms: float | None
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
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=120))
    _thread: threading.Thread | None = None
    _stop: threading.Event | None = None

    @property
    def beats_csv(self) -> Path:
        return self.archive_root / self.session_id / f"{self.session_id}_beats.csv"

    @property
    def local_beats_csv(self) -> Path:
        return self.capture_root / self.session_id / f"{self.session_id}_beats.csv"

    @property
    def metadata_json(self) -> Path:
        return self.archive_root / self.session_id / f"{self.session_id}_beats.json"

    @property
    def local_metadata_json(self) -> Path:
        return self.capture_root / self.session_id / f"{self.session_id}_beats.json"

    def drain(self, handler: Callable[[CNAPBeatEvent], None]) -> None:
        while True:
            try:
                handler(self.events.get_nowait())
            except Empty:
                return

    def diagnostic_summary(self) -> str:
        exit_code = self.process.poll()
        lines = [f"process_exit_code={exit_code}"]
        if self.local_beats_csv.exists():
            lines.append(f"local_beats_csv={self.local_beats_csv}")
        if self.beats_csv.exists():
            lines.append(f"archive_beats_csv={self.beats_csv}")
        if self.log_lines:
            lines.append("recent_cnap_output:")
            lines.extend(f"  {line}" for line in self.log_lines)
        return "\n".join(lines)

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=1)


def _pump_output(
    process: subprocess.Popen[str],
    queue: Queue[CNAPBeatEvent],
    log_lines: deque[str],
    stop: threading.Event,
) -> None:
    assert process.stdout is not None
    pattern = re.compile(
        r"\[beat\].*now=(?P<now>\S+)\s+t=(?P<elapsed>[0-9.]+)s #(?P<beat>\d+).*"
        r"Sys=(?P<sys>[0-9.]+).*Mean=(?P<mean>[0-9.]+).*Dia=(?P<dia>[0-9.]+).*HR=(?P<hr>[0-9.]+)"
    )
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log_lines.append(line)
        if stop.is_set():
            break
        match = pattern.search(line)
        if not match:
            continue
        timestamp_ms = None
        try:
            timestamp_ms = datetime.fromisoformat(match.group("now")).timestamp() * 1000.0
        except ValueError:
            timestamp_ms = None
        queue.put(
            CNAPBeatEvent(
                timestamp_ms=timestamp_ms,
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
    log_lines: deque[str] = deque(maxlen=120)
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
    thread = threading.Thread(target=_pump_output, args=(process, queue, log_lines, stop), daemon=True)
    thread.start()
    return CNAPCapture(
        process=process,
        session_id=session_id,
        capture_root=repo_root / "CNAP" / "captures",
        archive_root=archive_root,
        events=queue,
        log_lines=log_lines,
        _thread=thread,
        _stop=stop,
    )


def resolve_cnap_beats_csv(
    repo_root: Path,
    session_id: str,
    fallback_events: list[CNAPBeatEvent] | None = None,
) -> Path | None:
    archive_csv = repo_root / "Analysis" / "Data" / "pdp" / "realtime_aux" / session_id / f"{session_id}_beats.csv"
    archive_json = archive_csv.with_suffix(".json")
    local_csv = repo_root / "CNAP" / "captures" / session_id / f"{session_id}_beats.csv"
    local_json = local_csv.with_suffix(".json")

    if archive_csv.exists():
        return archive_csv

    if local_csv.exists():
        archive_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_csv, archive_csv)
        if local_json.exists():
            shutil.copy2(local_json, archive_json)
        return archive_csv

    if fallback_events:
        written = write_fallback_beats_csv(archive_csv, fallback_events)
        if written is not None:
            return written

    return None


def write_fallback_beats_csv(destination: Path, events: list[CNAPBeatEvent]) -> Path | None:
    valid_events = [event for event in events if event.timestamp_ms is not None]
    if not valid_events:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    started_iso = datetime.fromtimestamp(valid_events[0].timestamp_ms / 1000.0).astimezone().isoformat(timespec="milliseconds")
    fieldnames = [
        "開始時刻",
        "記録開始時刻",
        "現在時刻",
        "epoch_ns",
        "monotonic_ns",
        "経過時間",
        "計測回数",
        "BP waveform Raw",
        "BP waveform [mmHg]",
        "MAP Raw",
        "MAP [mmHg]",
        "CO Raw",
        "CO [L/min]",
        "Beat Sys [mmHg]",
        "Beat Mean [mmHg]",
        "Beat Dia [mmHg]",
        "Beat HR [bpm]",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in valid_events:
            timestamp_iso = datetime.fromtimestamp(event.timestamp_ms / 1000.0).astimezone().isoformat(timespec="milliseconds")
            writer.writerow(
                {
                    "開始時刻": started_iso,
                    "記録開始時刻": started_iso,
                    "現在時刻": timestamp_iso,
                    "epoch_ns": int(round(event.timestamp_ms * 1_000_000.0)),
                    "monotonic_ns": "",
                    "経過時間": f"{event.elapsed_time_s:.9f}",
                    "計測回数": event.beat_index,
                    "BP waveform Raw": "",
                    "BP waveform [mmHg]": "",
                    "MAP Raw": "",
                    "MAP [mmHg]": f"{event.mean:.9f}",
                    "CO Raw": "",
                    "CO [L/min]": "",
                    "Beat Sys [mmHg]": f"{event.systolic:.9f}",
                    "Beat Mean [mmHg]": f"{event.mean:.9f}",
                    "Beat Dia [mmHg]": f"{event.diastolic:.9f}",
                    "Beat HR [bpm]": f"{event.heart_rate:.9f}",
                }
            )
    return destination


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
