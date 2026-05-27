from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .common import discover_lumerical, ensure_first_run_config


def _cpu_range(threads: int) -> str:
    return f"0-{threads - 1}" if threads > 1 else "0"


def _run_dir(tag: str, base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{timestamp}_{tag}"


def _ensure_dir(path: Path) -> OSError | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return exc
    return None


def _default_output_base() -> tuple[Path, str | None]:
    preferred = Path(os.environ.get("EIDL_LUMERICAL_DATA_ROOT", "/Lumerical_data")).expanduser()
    exc = _ensure_dir(preferred)
    if exc is None:
        return preferred, None

    fallback = Path.home() / "Lumerical_data"
    fallback_exc = _ensure_dir(fallback)
    if fallback_exc is None:
        return fallback, (
            f"default output root {preferred} is not writable ({exc}); "
            f"using {fallback}"
        )

    return preferred, (
        f"default output root {preferred} is not writable ({exc}); "
        f"fallback {fallback} also failed ({fallback_exc})"
    )


def _command(script: Path, threads: int, use_taskset: bool) -> list[str]:
    python = Path(sys.executable)
    cmd = [str(python), "-u", str(script)]
    if use_taskset:
        taskset = shutil_which("taskset")
        if taskset:
            return [taskset, "-c", _cpu_range(threads), *cmd]
        print("  warning                   : taskset not found; running without CPU affinity")
    return cmd


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _write_info(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _prepend_pythonpath(env: dict[str, str], paths: list[Path]) -> None:
    existing = env.get("PYTHONPATH", "")
    parts = [str(path) for path in paths if path.exists()]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def _stream_process(cmd: list[str], env: dict[str, str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run",
        description="Run a Lumerical simulation script in the EIDL-Lumapi environment.",
    )
    parser.add_argument("script", help="Python simulation script to run.")
    parser.add_argument("-th", "--threads", type=int, required=True, help="CPU threads assigned to Lumerical FDTD.")
    parser.add_argument("-GPU", "--gpu", type=int, required=True, help="Physical GPU index to assign.")
    parser.add_argument("--tag", help="Optional profiling result tag.")
    parser.add_argument("--desc", help="Optional run description.")
    parser.add_argument("--no-taskset", action="store_true", help="Do not pin the Python process to CPU cores.")
    parser.add_argument(
        "--outdir",
        help=(
            "Directory for Lumerical byproducts and run logs. "
            "Default: /Lumerical_data if writable, otherwise ~/Lumerical_data."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command and environment without running.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable msopt resource profiling and write profiling_results under the run directory.",
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=None,
        help="Profiling sample interval in seconds. Default: 0.5.",
    )
    args = parser.parse_args(argv)
    ensure_first_run_config(verbose=True)

    if args.threads < 1:
        parser.error("-th/--threads must be >= 1")

    launch_cwd = Path.cwd()
    script = Path(args.script).expanduser()
    if not script.exists():
        parser.error(f"script not found: {script}")
    script = script.resolve()
    script_dir = script.parent

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["LUMERICAL_SESSION_GPU_DEVICE"] = f"GPU {args.gpu}"
    env["FDTD_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["NUMEXPR_NUM_THREADS"] = str(args.threads)
    env["QT_QPA_PLATFORM"] = env.get("QT_QPA_PLATFORM", "offscreen")
    env["EIDL_ORIGINAL_CWD"] = str(launch_cwd)
    env["EIDL_SCRIPT_DIR"] = str(script_dir)
    if args.profile:
        env["MSOPT_PROFILE"] = "1"
        env["EIDL_PROFILE"] = "1"
    if args.profile_interval is not None:
        env["MSOPT_PROFILE_INTERVAL"] = str(args.profile_interval)

    if not args.no_taskset:
        env["TASKSET_CPUS"] = _cpu_range(args.threads)

    install = discover_lumerical()
    if install.lumapi_dir:
        env["LUMERICAL_PYTHONPATH"] = env.get("LUMERICAL_PYTHONPATH", str(install.lumapi_dir))
    if install.bin_dir:
        env["LUMERICAL_BIN_DIR"] = env.get("LUMERICAL_BIN_DIR", str(install.bin_dir))
        env["PATH"] = f"{install.bin_dir}:{env.get('PATH', '')}"
    if install.root:
        env["LUMERICAL_ROOT"] = env.get("LUMERICAL_ROOT", str(install.root))

    tag = args.tag or f"{script.stem}_gpu{args.gpu}_th{args.threads}"
    desc = args.desc or f"GPU {args.gpu}, FDTD threads {args.threads}"
    cmd = _command(script, args.threads, not args.no_taskset)
    if args.outdir:
        output_base = Path(args.outdir).expanduser()
        output_warning = None
    else:
        output_base, output_warning = _default_output_base()
    run_dir = _run_dir(tag, output_base)
    log_path = run_dir / f"{script.stem}_output.log"
    env["EIDL_RUN_DIR"] = str(run_dir)
    env["EIDL_LUMERICAL_DATA_DIR"] = str(output_base)
    _prepend_pythonpath(env, [launch_cwd, script_dir])

    print("EIDL-Lumapi run")
    print(f"  script                    : {script}")
    print(f"  GPU                       : {args.gpu}")
    print(f"  FDTD threads              : {args.threads}")
    print(f"  CPU affinity              : {env.get('TASKSET_CPUS', '(disabled)')}")
    print(f"  LUMERICAL_ROOT            : {env.get('LUMERICAL_ROOT', '(not detected)')}")
    print(f"  LUMERICAL_PYTHONPATH      : {env.get('LUMERICAL_PYTHONPATH', '(not detected)')}")
    print(f"  profiling                 : {'on' if env.get('MSOPT_PROFILE') == '1' else 'off'}")
    if output_warning:
        print(f"  output warning            : {output_warning}")
    print(f"  execution cwd             : {run_dir}")
    print(f"  output directory          : {run_dir}")
    print(f"  command                   : {' '.join(cmd)}")

    if args.dry_run:
        return 0

    exc = _ensure_dir(run_dir)
    if exc is not None:
        print(f"Failed to create Lumerical data directory: {run_dir}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print("Use --outdir /path/you/can/write.", file=sys.stderr)
        return 2
    _write_info(
        run_dir / "info.txt",
        [
            f"Run: {run_dir.name}",
            f"Description: {desc}",
            f"Script: {script}",
            f"Launch cwd: {launch_cwd}",
            f"Execution cwd: {run_dir}",
            f"GPU: {args.gpu}",
            f"FDTD threads: {args.threads}",
            f"CPU affinity: {env.get('TASKSET_CPUS', '(disabled)')}",
            f"LUMERICAL_ROOT: {env.get('LUMERICAL_ROOT', '(not detected)')}",
            f"LUMERICAL_PYTHONPATH: {env.get('LUMERICAL_PYTHONPATH', '(not detected)')}",
            f"Command: {' '.join(cmd)}",
            f"Output log: {log_path}",
            f"Profiling: {'on' if env.get('MSOPT_PROFILE') == '1' else 'off'}",
            f"Profiling output: {run_dir / 'profiling_results'}",
        ],
    )
    return_code = _stream_process(cmd, env, log_path, run_dir)
    with (run_dir / "info.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"Exit code: {return_code}\n")
        handle.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
    print(f"\nEIDL-Lumapi run finished with exit code {return_code}")
    print(f"Log: {log_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
