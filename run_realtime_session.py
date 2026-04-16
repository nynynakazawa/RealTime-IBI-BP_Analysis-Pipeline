#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime

from current_direction import REALTIME_SESSIONS_ROOT, REPO_ROOT
from realtime_pipeline.android_bridge import (
    AndroidBeatEvent,
    AndroidLogcatMonitor,
    clear_logcat,
    ensure_device_ready,
    list_remote_session_files,
    phone_is_ready,
    pull_session_files,
    start_session,
    stop_session,
)
from realtime_pipeline.cnap_bridge import CNAPBeatEvent, cnap_is_ready, start_cnap_capture
from realtime_pipeline.cnap_bridge import resolve_cnap_beats_csv
from realtime_pipeline.evaluate_session import (
    evaluate_merged_session,
    generate_session_plots,
    write_session_report,
)
from realtime_pipeline.merge_session import merge_session_data
from realtime_pipeline.session_filtered_input import list_realtime_session_dirs
import pandas as pd


LEGACY_EXPERIMENTAL_FILES = (
    "session_evaluation_summary_experimental.csv",
    "session_evaluation_summary_experimental.json",
    "session_evaluation_summary_experimental_app_export_backup.csv",
    "session_evaluation_summary_experimental_app_export_backup.json",
    "session_evaluation_summary_experimental_meta.json",
    "session_evaluation_predictions_experimental_repaired.csv",
)


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


def cleanup_legacy_experimental_outputs(evaluation_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for name in LEGACY_EXPERIMENTAL_FILES:
        path = evaluation_dir / name
        if not path.exists():
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNAP + smartphone realtime pipeline (active methods: RTBP / SinBP_D / SinBP_M)"
    )
    parser.add_argument("--subject-id", default="")
    parser.add_argument("--session-number", type=int, default=1)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--recover-session-id", default="")
    # Baseline-family experimental repair flow is intentionally disabled in normal runs.
    parser.add_argument("--rerun-existing-evaluations", action="store_true")
    parser.add_argument("--target-session", default="")
    parser.add_argument(
        "--past",
        action="store_true",
        help="Include sessions under Analysis/Data/realtime_sessions/past/ when scanning existing sessions.",
    )
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
    rtbp_sbp = rtbp.get("sbp_process", rtbp.get("sbp", 0))
    rtbp_dbp = rtbp.get("dbp_process", rtbp.get("dbp", 0))
    sin_d_sbp = sin_d.get("sbp_process", sin_d.get("sbp", 0))
    sin_d_dbp = sin_d.get("dbp_process", sin_d.get("dbp", 0))
    sin_m_sbp = sin_m.get("sbp_process", sin_m.get("sbp", 0))
    sin_m_dbp = sin_m.get("dbp_process", sin_m.get("dbp", 0))
    return (
        f"[phone] t={event.elapsed_time_s:6.2f}s beat={event.beat_index:03d} "
        f"RTBP={rtbp_sbp:6.1f}/{rtbp_dbp:5.1f} "
        f"SinD={sin_d_sbp:6.1f}/{sin_d_dbp:5.1f} "
        f"SinM={sin_m_sbp:6.1f}/{sin_m_dbp:5.1f}"
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
    rtbp_sbp = rtbp.get("sbp_process", rtbp.get("sbp", 0))
    rtbp_dbp = rtbp.get("dbp_process", rtbp.get("dbp", 0))
    sin_d_sbp = sin_d.get("sbp_process", sin_d.get("sbp", 0))
    sin_d_dbp = sin_d.get("dbp_process", sin_d.get("dbp", 0))
    sin_m_sbp = sin_m.get("sbp_process", sin_m.get("sbp", 0))
    sin_m_dbp = sin_m.get("dbp_process", sin_m.get("dbp", 0))
    dt_s = (
        (phone.timestamp_ms - float(cnap.timestamp_ms)) / 1000.0
        if cnap.timestamp_ms is not None and phone.timestamp_ms > 0
        else phone.elapsed_time_s - cnap.elapsed_time_s
    )
    return (
        f"[sync ] t_phone={phone.elapsed_time_s:6.2f}s beat={phone.beat_index:03d} "
        f"CNAP={cnap.systolic:6.1f}/{cnap.diastolic:5.1f} "
        f"RTBP={rtbp_sbp:6.1f}/{rtbp_dbp:5.1f} "
        f"SinD={sin_d_sbp:6.1f}/{sin_d_dbp:5.1f} "
        f"SinM={sin_m_sbp:6.1f}/{sin_m_dbp:5.1f} "
        f"dt={dt_s:+5.2f}s"
    )


def select_nearest_cnap(phone: AndroidBeatEvent, cnap_events: deque[CNAPBeatEvent]) -> CNAPBeatEvent | None:
    if not cnap_events:
        return None
    if phone.timestamp_ms <= 0:
        return cnap_events[-1]
    candidates = [event for event in cnap_events if event.timestamp_ms is not None]
    if not candidates:
        return cnap_events[-1]
    return min(candidates, key=lambda event: abs(phone.timestamp_ms - float(event.timestamp_ms)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kill_stale_session_processes(os.getpid())
    repo_root = REPO_ROOT
    sessions_root = REALTIME_SESSIONS_ROOT

    if args.rerun_existing_evaluations:
        target_session = args.target_session.strip()
        session_dirs = list_realtime_session_dirs(sessions_root, include_past=args.past, require_artifacts=False)
        rerun_count = 0
        for session_dir in session_dirs:
            session_id = session_dir.name
            if target_session and session_id != target_session:
                continue
            merged_csv = session_dir / f"{session_id}_merged.csv"
            if not merged_csv.exists():
                continue
            merged_df = pd.read_csv(merged_csv)
            eval_dir = session_dir / "evaluation"
            summary_csv, summary_json = evaluate_merged_session(merged_df, eval_dir)
            summary_df = pd.read_csv(summary_csv)
            plot_paths = generate_session_plots(merged_df, eval_dir)
            report_path = write_session_report(merged_df, summary_df, eval_dir, plot_paths)
            cleanup_legacy_experimental_outputs(eval_dir)
            rerun_count += 1
            print(f"reran -> {session_id}")
            print(f"evaluation summary -> {summary_csv}")
            print(f"evaluation json -> {summary_json}")
            print(f"session report -> {report_path}")
        if rerun_count == 0:
            print("no existing sessions were rerun")
            return 1
        print(f"reran evaluations: {rerun_count}")
        return 0

    phone_ok, phone_status = phone_is_ready()
    cnap_ok, cnap_status = cnap_is_ready(repo_root)
    if not phone_ok:
        print(f"phone check failed: {phone_status}")
        return 1
    if not args.recover_session_id and not cnap_ok:
        print(f"cnap check failed: {cnap_status}")
        return 1

    if args.recover_session_id:
        session_id = args.recover_session_id.strip()
        subject_id = args.subject_id or session_id.split("_")[0]
        session_number = args.session_number
    else:
        subject_id = prompt_if_missing(args.subject_id, "subject_id")
        session_number = int(prompt_if_missing(str(args.session_number), "session_number", str(args.session_number)))
        default_session_id = args.session_id or f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_id = prompt_if_missing(args.session_id, "session_id", default_session_id)

    session_root = sessions_root / session_id
    smartphone_dir = session_root / "smartphone"
    merged_csv = session_root / f"{session_id}_merged.csv"
    eval_dir = session_root / "evaluation"

    if args.recover_session_id:
        pulled = pull_session_files(session_id, smartphone_dir)
        training_csv = smartphone_dir / f"{session_id}_Training_Data.csv"
        if training_csv not in pulled and not training_csv.exists():
            remote_files = list_remote_session_files(session_id)
            print(
                "recover failed: smartphone training csv was not pulled.\n"
                f"expected: {training_csv}\n"
                "available remote files:\n"
                + ("\n".join(f"  {path}" for path in remote_files) if remote_files else "  (none)")
            )
            return 1
        cnap_beats_csv = resolve_cnap_beats_csv(repo_root, session_id)
        if cnap_beats_csv is None:
            expected = repo_root / "Analysis" / "Data" / "pdp" / "realtime_aux" / session_id / f"{session_id}_beats.csv"
            print(f"recover failed: CNAP beats csv was not found: {expected}")
            return 1
        merged_df = merge_session_data(training_csv, cnap_beats_csv, merged_csv)
        summary_csv, summary_json = evaluate_merged_session(merged_df, eval_dir)
        summary_df = pd.read_csv(summary_csv)
        plot_paths = generate_session_plots(merged_df, eval_dir)
        report_path = write_session_report(merged_df, summary_df, eval_dir, plot_paths)
        removed_legacy = cleanup_legacy_experimental_outputs(eval_dir)
        print(f"recovered smartphone files -> {smartphone_dir}")
        print(f"cnap beats -> {cnap_beats_csv}")
        print(f"merged csv -> {merged_csv}")
        print(f"evaluation summary -> {summary_csv}")
        print(f"evaluation json -> {summary_json}")
        if removed_legacy:
            print(f"removed legacy experimental files: {len(removed_legacy)}")
        print(f"session report -> {report_path}")
        return 0

    try:
        ensure_device_ready()
        clear_logcat()
        monitor = AndroidLogcatMonitor()
        monitor.start()

        cnap = start_cnap_capture(repo_root, session_id)
        start_session(session_id=session_id, subject_id=subject_id, session_number=session_number, mode=args.mode)
        recent_cnap: deque[CNAPBeatEvent] = deque(maxlen=64)
        all_cnap: list[CNAPBeatEvent] = []

        print(f"session_id={session_id}")
        print("recording started. Press Ctrl+C to stop.")

        while True:
            def on_cnap(event: CNAPBeatEvent) -> None:
                recent_cnap.append(event)
                all_cnap.append(event)

            def on_phone(event: AndroidBeatEvent) -> None:
                print(render_combined_event(event, select_nearest_cnap(event, recent_cnap)), flush=True)

            cnap.drain(on_cnap)
            monitor.drain(on_phone)
            if cnap.process.poll() is not None and not all_cnap:
                raise RuntimeError(
                    "CNAP capture exited before producing beats.\n"
                    + cnap.diagnostic_summary()
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping...")
    except Exception as exc:
        print(f"session failed: {exc}")
        return 1

    for _ in range(2):
        try:
            stop_session(session_id)
        except Exception:
            pass
        time.sleep(2.0)
    cnap.stop()
    cnap.drain(lambda event: all_cnap.append(event))
    monitor.drain(lambda event: print(render_combined_event(event, select_nearest_cnap(event, recent_cnap)), flush=True))
    monitor.stop()

    pulled = pull_session_files(session_id, smartphone_dir)
    training_csv = smartphone_dir / f"{session_id}_Training_Data.csv"
    if training_csv not in pulled and not training_csv.exists():
        remote_files = list_remote_session_files(session_id)
        print(
            "session stopped, but smartphone training csv was not pulled.\n"
            f"expected: {training_csv}\n"
            "checked remote paths:\n"
            f"  /sdcard/Download/{session_id}_Training_Data.csv\n"
            f"  /sdcard/Download/PC_Sync/Analysis/Data/Smartphone/{session_id}/{session_id}_Training_Data.csv\n"
            "available remote files:\n"
            + ("\n".join(f"  {path}" for path in remote_files) if remote_files else "  (none)")
        )
        return 1
    cnap_beats_csv = resolve_cnap_beats_csv(repo_root, session_id, fallback_events=all_cnap)
    if cnap_beats_csv is None:
        print(
            f"session stopped, but CNAP beats csv was not created: {cnap.beats_csv}\n"
            + cnap.diagnostic_summary()
        )
        return 1

    merged_df = merge_session_data(training_csv, cnap_beats_csv, merged_csv)
    summary_csv, summary_json = evaluate_merged_session(merged_df, eval_dir)
    summary_df = pd.read_csv(summary_csv)
    plot_paths = generate_session_plots(merged_df, eval_dir)
    report_path = write_session_report(merged_df, summary_df, eval_dir, plot_paths)
    removed_legacy = cleanup_legacy_experimental_outputs(eval_dir)

    print(f"pulled smartphone files -> {smartphone_dir}")
    print(f"cnap beats -> {cnap_beats_csv}")
    print(f"merged csv -> {merged_csv}")
    print(f"evaluation summary -> {summary_csv}")
    print(f"evaluation json -> {summary_json}")
    if removed_legacy:
        print(f"removed legacy experimental files: {len(removed_legacy)}")
    print(f"session report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
