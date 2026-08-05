from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runbg",
        description=(
            "Start the existing `run` command in a detached background process "
            "that keeps running after the current terminal exits."
        ),
        epilog=(
            "All other arguments are passed through to `run`. "
            "Example: runbg script_name.py -th 8 -GPU 2"
        ),
    )
    parser.add_argument(
        "--bg-log",
        metavar="PATH",
        help=(
            "File for detached launcher output. Default: "
            "~/.cache/EIDL-Lumapi/runbg/<timestamp>_<script>.log"
        ),
    )
    parser.add_argument(
        "--bg-dry-run",
        action="store_true",
        help="Print the detached command without launching it.",
    )
    return parser


def _safe_label(text: str) -> str:
    label = Path(text).expanduser().stem or "run"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in label)
    return safe.strip("._") or "run"


def _script_label(run_args: list[str]) -> str:
    options_with_values = {
        "-th",
        "--threads",
        "-GPU",
        "--gpu",
        "--tag",
        "--desc",
        "--outdir",
        "--profile-interval",
    }
    long_options_with_values = {
        "--threads",
        "--gpu",
        "--tag",
        "--desc",
        "--outdir",
        "--profile-interval",
    }
    skip_next = False
    for arg in run_args:
        if skip_next:
            skip_next = False
            continue
        if arg in options_with_values:
            skip_next = True
            continue
        if any(arg.startswith(f"{option}=") for option in long_options_with_values):
            continue
        if arg.startswith("-"):
            continue
        return _safe_label(arg)
    return "run"


def _default_log_path(run_args: list[str]) -> Path:
    root = Path(
        os.environ.get(
            "EIDL_RUNBG_LOG_DIR",
            str(Path.home() / ".cache" / "EIDL-Lumapi" / "runbg"),
        )
    ).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{timestamp}_{_script_label(run_args)}.log"


def _prepare_log_path(path: Path, explicit: bool) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        if explicit:
            raise
        fallback = Path(tempfile.gettempdir()) / f"EIDL-Lumapi-runbg-{os.getuid()}" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback
    return path


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, run_args = parser.parse_known_args(argv)
    if not run_args:
        parser.print_usage(sys.stderr)
        print("runbg: error: missing arguments for `run`", file=sys.stderr)
        print("example: runbg script_name.py -th 8 -GPU 2", file=sys.stderr)
        return 2

    cmd = [sys.executable, "-u", "-m", "msopt.cli.run", *run_args]
    explicit_log = args.bg_log is not None
    raw_log_path = Path(args.bg_log).expanduser() if explicit_log else _default_log_path(run_args)
    try:
        log_path = _prepare_log_path(raw_log_path, explicit_log)
    except OSError as exc:
        print(f"runbg: failed to create log directory for {raw_log_path}: {exc}", file=sys.stderr)
        return 2

    cwd = Path.cwd()
    env = os.environ.copy()
    env["EIDL_RUNBG"] = "1"
    env["EIDL_RUNBG_LOG"] = str(log_path)
    command_text = _quote_cmd(cmd)

    if args.bg_dry_run:
        print("EIDL-Lumapi runbg dry run")
        print(f"  cwd     : {cwd}")
        print(f"  log     : {log_path}")
        print(f"  command : {command_text}")
        return 0

    started = datetime.now().isoformat(timespec="seconds")
    try:
        with log_path.open("ab") as log:
            prelude = (
                "EIDL-Lumapi runbg launcher\n"
                f"Started: {started}\n"
                f"CWD: {cwd}\n"
                f"Command: {command_text}\n\n"
            )
            log.write(prelude.encode("utf-8", errors="replace"))
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        print(f"runbg: failed to launch background run: {exc}", file=sys.stderr)
        return 2

    print("EIDL-Lumapi runbg")
    print(f"  PID       : {proc.pid}")
    print(f"  log       : {log_path}")
    print("  lifecycle : same as run (Ongoing -> Done/Failed)")
    print(f"  follow    : tail -f {shlex.quote(str(log_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
