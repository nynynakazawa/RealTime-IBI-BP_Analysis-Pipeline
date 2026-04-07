#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime

from realtime_pipeline.android_bridge import (
    AndroidBeatEvent,
    AndroidLogcatMonitor,
    clear_logcat,
    ensure_device_ready,
    phone_is_ready,
    pull_session_files,
    start_session,
    stop_session,
)
from realtime_pipeline.cnap_bridge import CNAPBeatEvent, cnap_is_ready, start_cnap_capture
from realtime_pipeline.evaluate_session import (
    evaluate_merged_session,
    generate_session_plots,
    write_session_report,
)
from realtime_pipeline.merge_session import merge_session_data
import pandas as pd


def kill_stale_session_processes(current_pid: int) -> None:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return

    current_script = str(Path(__file__).resolve())
    realtime_capture_script = str((Path(__file__).resolve().parent.parent / "CNAP" / "realtime_capture.py").resolve())
    stale_pids: list[int] = []

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if current_script in command or realtime_capture_script in command:
            stale_pids.append(pid)

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    if stale_pids:
        time.sleep(1.0)

    for pid in stale_pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNAP + smartphone realtime pipeline")
    parser.add_argument("--subject-id", default="")
    parser.add_argument("--session-number", type=int, default=1)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--session-id", default="")
    return parser


def prompt_if_missing(value: str, label: str, default: str = "") -> str:
    if value:
        return value
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    entered = input(prompt).strip()
    return entered or default


def render_event(event: AndroidBeatEvent) -> str:
    rtbp = event.rtbp
    sin_d = event.sinbp_d
    sin_m = event.sinbp_m
    return (
        f"[phone] t={event.elapsed_time_s:6.2f}s beat={event.beat_index:03d} "
        f"RTBP={rtbp.get('sbp', 0):6.1f}/{rtbp.get('dbp', 0):5.1f} "
        f"SinD={sin_d.get('sbp', 0):6.1f}/{sin_d.get('dbp', 0):5.1f} "
        f"SinM={sin_m.get('sbp', 0):6.1f}/{sin_m.get('dbp', 0):5.1f}"
    )


def render_cnap_event(event: CNAPBeatEvent) -> str:
    return (
        f"[cnap ] t={event.elapsed_time_s:6.2f}s beat={event.beat_index:03d} "
        f"CNAP={event.systolic:6.1f}/{event.diastolic:5.1f} "
        f"MAP={event.mean:5.1f} HR={event.heart_rate:5.1f}"
    )


def render_combined_event(phone: AndroidBeatEvent, cnap: CNAPBeatEvent | None) -> str:
    if cnap is None:
        return render_event(phone) + " CNAP= ---/---"
    rtbp = phone.rtbp
    sin_d = phone.sinbp_d
    sin_m = phone.sinbp_m
    return (
        f"[sync ] t_phone={phone.elapsed_time_s:6.2f}s beat={phone.beat_index:03d} "
        f"CNAP={cnap.systolic:6.1f}/{cnap.diastolic:5.1f} "
        f"RTBP={rtbp.get('sbp', 0):6.1f}/{rtbp.get('dbp', 0):5.1f} "
        f"SinD={sin_d.get('sbp', 0):6.1f}/{sin_d.get('dbp', 0):5.1f} "
        f"SinM={sin_m.get('sbp', 0):6.1f}/{sin_m.get('dbp', 0):5.1f} "
        f"dt={phone.elapsed_time_s - cnap.elapsed_time_s:+5.2f}s"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kill_stale_session_processes(os.getpid())
    repo_root = Path(__file__).resolve().parent.parent

    phone_ok, phone_status = phone_is_ready()
    cnap_ok, cnap_status = cnap_is_ready(repo_root)
    if not phone_ok:
        print(f"phone check failed: {phone_status}")
        return 1
    if not cnap_ok:
        print(f"cnap check failed: {cnap_status}")
        return 1

    subject_id = prompt_if_missing(args.subject_id, "subject_id")
    session_number = int(prompt_if_missing(str(args.session_number), "session_number", str(args.session_number)))
    default_session_id = args.session_id or f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_id = prompt_if_missing(args.session_id, "session_id", default_session_id)

    session_root = repo_root / "Analysis" / "Data" / "realtime_sessions" / session_id
    smartphone_dir = session_root / "smartphone"
    merged_csv = session_root / f"{session_id}_merged.csv"
    eval_dir = session_root / "evaluation"

    try:
        ensure_device_ready()
        clear_logcat()
        monitor = AndroidLogcatMonitor()
        monitor.start()

        cnap = start_cnap_capture(repo_root, session_id)
        start_session(session_id=session_id, subject_id=subject_id, session_number=session_number, mode=args.mode)
        latest_cnap: CNAPBeatEvent | None = None

        print(f"session_id={session_id}")
        print("recording started. Press Ctrl+C to stop.")

        while True:
            def on_cnap(event: CNAPBeatEvent) -> None:
                nonlocal latest_cnap
                latest_cnap = event

            def on_phone(event: AndroidBeatEvent) -> None:
                print(render_combined_event(event, latest_cnap), flush=True)

            cnap.drain(on_cnap)
            monitor.drain(on_phone)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping...")
    except Exception as exc:
        print(f"session failed: {exc}")
        return 1

    stop_session()
    time.sleep(4.0)
    cnap.stop()
    monitor.drain(lambda event: print(render_combined_event(event, latest_cnap), flush=True))
    monitor.stop()

    pulled = pull_session_files(session_id, smartphone_dir)
    training_csv = smartphone_dir / f"{session_id}_Training_Data.csv"
    if training_csv not in pulled and not training_csv.exists():
        print(
            "session stopped, but smartphone training csv was not pulled.\n"
            f"expected: {training_csv}\n"
            "checked remote paths:\n"
            f"  /sdcard/Download/{session_id}_Training_Data.csv\n"
            f"  /sdcard/Download/PC_Sync/Analysis/Data/Smartphone/{session_id}/{session_id}_Training_Data.csv"
        )
        return 1
    if not cnap.beats_csv.exists():
        print(f"session stopped, but CNAP beats csv was not created: {cnap.beats_csv}")
        return 1

    merged_df = merge_session_data(training_csv, cnap.beats_csv, merged_csv)
    summary_csv, summary_json = evaluate_merged_session(merged_df, eval_dir)
    summary_df = pd.read_csv(summary_csv)
    plot_paths = generate_session_plots(merged_df, eval_dir)
    report_path = write_session_report(merged_df, summary_df, eval_dir, plot_paths)

    print(f"pulled smartphone files -> {smartphone_dir}")
    print(f"cnap beats -> {cnap.beats_csv}")
    print(f"merged csv -> {merged_csv}")
    print(f"evaluation summary -> {summary_csv}")
    print(f"evaluation json -> {summary_json}")
    print(f"session report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
