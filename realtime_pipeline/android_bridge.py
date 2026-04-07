from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Callable


PACKAGE_NAME = "com.nakazawa.realtimeibibp"
ACTIVITY_NAME = f"{PACKAGE_NAME}/.MainActivity"
ACTION_START = f"{PACKAGE_NAME}.action.START_AUTOMATED_SESSION"
ACTION_STOP = f"{PACKAGE_NAME}.action.STOP_AUTOMATED_SESSION"


@dataclass(frozen=True)
class AndroidBeatEvent:
    raw: dict

    @property
    def beat_index(self) -> int:
        return int(self.raw.get("beat_index", 0))

    @property
    def elapsed_time_s(self) -> float:
        return float(self.raw.get("elapsed_time_s", 0.0))

    @property
    def rtbp(self) -> dict:
        return dict(self.raw.get("rtbp", {}))

    @property
    def sinbp_d(self) -> dict:
        return dict(self.raw.get("sinbp_d", {}))

    @property
    def sinbp_m(self) -> dict:
        return dict(self.raw.get("sinbp_m", {}))


def run_adb(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def adb_devices_output() -> str:
    result = run_adb("devices", "-l", check=False)
    return (result.stdout or "") + (result.stderr or "")


def phone_is_ready() -> tuple[bool, str]:
    run_adb("start-server", check=False)
    devices = adb_devices_output().strip()
    lines = [line.strip() for line in devices.splitlines() if line.strip()]
    for line in lines[1:]:
        if "\tdevice" in line or " device " in line:
            return True, line
    if len(lines) <= 1:
        return False, "phone is not detected by adb"
    return False, f"phone is not ready: {lines[-1]}"


def _normalize_adb_state(raw: str) -> str:
    return raw.strip().splitlines()[-1].strip() if raw.strip() else ""


def ensure_device_ready() -> None:
    run_adb("start-server", check=False)
    last_error = ""
    for _ in range(3):
        result = run_adb("get-state", check=False)
        state = _normalize_adb_state((result.stdout or "") + (result.stderr or ""))
        if result.returncode == 0 and state == "device":
            return
        last_error = state or f"exit_status={result.returncode}"
        if "offline" in adb_devices_output():
            run_adb("kill-server", check=False)
            time.sleep(0.5)
            run_adb("start-server", check=False)
        time.sleep(1.0)

    devices = adb_devices_output().strip()
    raise RuntimeError(
        "adb device is not ready. "
        "Check that the phone is unlocked, USB debugging is authorized, and `adb devices -l` shows `device`.\n"
        f"last_adb_state: {last_error}\n"
        f"adb_devices:\n{devices}"
    )


def clear_logcat() -> None:
    run_adb("logcat", "-c")


def start_session(session_id: str, subject_id: str, session_number: int, mode: int) -> None:
    run_adb(
        "shell",
        "am",
        "start",
        "-n",
        ACTIVITY_NAME,
        "-a",
        ACTION_START,
        "--es",
        "session_id",
        session_id,
        "--es",
        "subject_id",
        subject_id,
        "--ei",
        "session_number",
        str(session_number),
        "--ei",
        "mode",
        str(mode),
    )


def stop_session() -> None:
    run_adb(
        "shell",
        "am",
        "start",
        "-n",
        ACTIVITY_NAME,
        "-a",
        ACTION_STOP,
    )


class AndroidLogcatMonitor:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._queue: Queue[AndroidBeatEvent] = Queue()
        self._stop = threading.Event()

    def start(self) -> None:
        self._process = subprocess.Popen(
            ["adb", "logcat", "RealtimeSession:I", "*:S"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        pattern = re.compile(r"RealtimeSession\s*:\s*(\{.*\})")
        for line in self._process.stdout:
            if self._stop.is_set():
                break
            match = pattern.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "bp_beat":
                continue
            self._queue.put(AndroidBeatEvent(payload))

    def drain(self, handler: Callable[[AndroidBeatEvent], None]) -> None:
        while True:
            try:
                handler(self._queue.get_nowait())
            except Empty:
                return

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=1)


def pull_session_files(session_id: str, destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffixes = [
        "_Training_Data.csv",
        "_RTBP.csv",
        "_SinBP_D.csv",
        "_SinBP_M.csv",
        "_Wave_Data.csv",
        "_元データ.csv",
    ]
    remote_candidates = [
        "/sdcard/Download",
        f"/sdcard/Download/PC_Sync/Analysis/Data/Smartphone/{session_id}",
    ]
    pulled: list[Path] = []
    for _ in range(8):
        pulled.clear()
        for suffix in suffixes:
            local_path = destination_dir / f"{session_id}{suffix}"
            if local_path.exists():
                local_path.unlink()
            for remote_dir in remote_candidates:
                remote_path = f"{remote_dir}/{session_id}{suffix}"
                result = run_adb("pull", remote_path, str(local_path), check=False)
                if result.returncode == 0 and local_path.exists():
                    pulled.append(local_path)
                    break
        if any(path.name.endswith("_Training_Data.csv") for path in pulled):
            break
        time.sleep(1.0)
    return pulled
