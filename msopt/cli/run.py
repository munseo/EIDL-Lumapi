from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import socket
from datetime import datetime
from pathlib import Path

from .common import discover_lumerical, ensure_first_run_config


def _parse_cpu_list(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            cpus.update(range(int(start_text), int(end_text) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _format_cpu_list(cpus: list[int] | set[int]) -> str:
    ordered = sorted(cpus)
    if not ordered:
        return ""
    ranges = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _read_proc_affinity(pid: int) -> set[int] | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(errors="replace").splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return _parse_cpu_list(line.split(":", 1)[1].strip())
    except OSError:
        return None
    return None


def _read_proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _relevant_lumerical_process(cmdline: str) -> bool:
    markers = (
        "EIDL-Lumapi/bin/run",
        "EIDL-Lumapi/applications/",
        "/opt/lumerical/",
        "fdtd-engine",
        "fdtd-solutions",
    )
    return any(marker in cmdline for marker in markers)


def _active_lumerical_cpu_usage(all_cpus: set[int]) -> tuple[set[int], list[tuple[int, str, str]]]:
    occupied: set[int] = set()
    details: list[tuple[int, str, str]] = []
    seen_affinity_specs: set[str] = set()
    current_pid = os.getpid()
    current_uid = os.getuid()
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        pid = int(proc_path.name)
        if pid == current_pid:
            continue
        try:
            if proc_path.stat().st_uid != current_uid:
                continue
        except OSError:
            continue
        cmdline = _read_proc_cmdline(pid)
        if not cmdline or not _relevant_lumerical_process(cmdline):
            continue
        affinity = _read_proc_affinity(pid)
        if not affinity:
            continue
        # Ignore unrestricted wrapper/orphan processes; they do not represent
        # a deliberate thread allocation for a running simulation.
        if affinity >= all_cpus:
            continue
        occupied.update(affinity)
        affinity_spec = _format_cpu_list(affinity)
        if affinity_spec not in seen_affinity_specs:
            seen_affinity_specs.add(affinity_spec)
            details.append((pid, affinity_spec, cmdline[:140].strip()))
    return occupied, details


def _choose_cpu_affinity(threads: int) -> tuple[str, str | None, int]:
    total = os.cpu_count() or threads
    all_cpus = set(range(total))
    occupied, details = _active_lumerical_cpu_usage(all_cpus)
    free = sorted(all_cpus - occupied)
    allocatable = len(free)
    if allocatable >= threads:
        selected = free[:threads]
        return _format_cpu_list(selected), None, allocatable

    selected = free[:]
    for cpu in sorted(all_cpus):
        if len(selected) >= threads:
            break
        if cpu not in selected:
            selected.append(cpu)
    warning_lines = [
        f"requested {threads} threads but only {allocatable} non-overlapping CPU cores are currently available",
        f"allocatable non-overlap threads: {allocatable}",
        f"assigned CPUs with overlap: {_format_cpu_list(selected)}",
    ]
    if details:
        warning_lines.append("active CPU allocations:")
        for pid, cpus, cmdline in details[:8]:
            warning_lines.append(f"  pid {pid}: CPUs {cpus} | {cmdline}")
        if len(details) > 8:
            warning_lines.append(f"  ... {len(details) - 8} more processes")
    return _format_cpu_list(selected), "; ".join(warning_lines), allocatable


def _run_dir(tag: str, base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{timestamp}_{tag}"


def _ensure_dir(path: Path) -> OSError | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return exc
    return None


def _prepare_lifecycle_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    ongoing = base_dir / "Ongoing"
    done = base_dir / "Done"
    failed = base_dir / "Failed"
    for path in (ongoing, done, failed):
        path.mkdir(parents=True, exist_ok=True)
    return ongoing, done, failed


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not create unique destination for {path}")


def _move_run_dir(run_dir: Path, destination_root: Path) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(destination_root / run_dir.name)
    shutil.move(str(run_dir), str(destination))
    return destination


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


def _command(script: Path, cpu_affinity: str, use_taskset: bool) -> list[str]:
    python = Path(sys.executable)
    cmd = [str(python), "-u", str(script)]
    if use_taskset:
        taskset = shutil_which("taskset")
        if taskset:
            return [taskset, "-c", cpu_affinity, *cmd]
        print("  warning                   : taskset not found; running without CPU affinity")
    return cmd


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _query_gpu_uuids() -> dict[int, str]:
    nvidia_smi = shutil_which("nvidia-smi")
    if not nvidia_smi:
        return {}
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return {}
    uuids: dict[int, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            uuids[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return uuids


def _active_gpu_compute_jobs(gpu_index: int) -> list[str]:
    nvidia_smi = shutil_which("nvidia-smi")
    if not nvidia_smi:
        return []
    gpu_uuid = _query_gpu_uuids().get(gpu_index)
    if not gpu_uuid:
        return []
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return []
    jobs: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or parts[0] != gpu_uuid:
            continue
        jobs.append(f"pid={parts[1]}, proc={Path(parts[2]).name}, vram={parts[3]} MiB")
    return jobs


def _write_info(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _prepend_pythonpath(env: dict[str, str], paths: list[Path]) -> None:
    existing = env.get("PYTHONPATH", "")
    parts = [str(path) for path in paths if path.exists()]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_ansys_port_file(path: Path) -> tuple[int, int] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        port_text, pid_text = text.split(":", 1)
        port = int(port_text)
        pid = int(pid_text)
    except Exception:
        return None
    if port <= 0 or pid <= 0 or not _pid_is_alive(pid):
        return None
    return port, pid


def _lumerical_version_tag(root: str | None) -> str | None:
    if not root:
        return None
    name = Path(root).name
    if name.startswith("v") and name[1:].isdigit():
        return name[1:]
    digits = "".join(ch for ch in name if ch.isdigit())
    return digits or None


def _ensure_ansys_shared_port_link(env: dict[str, str]) -> str | None:
    """Attach normal HOME runs to an already running redirected Ansys LI daemon.

    Some Lumerical jobs redirect HOME to /tmp/tfln_lumerical_runtime_*.
    Ansys license sharing then writes the .ansyscl port file under that
    redirected HOME. A later normal run under /home/eidl may fail before
    checkout because it looks for the same sharing context under ~/.ansys.
    """
    version = _lumerical_version_tag(env.get("LUMERICAL_ROOT"))
    if not version:
        return None

    home = Path(env.get("HOME", str(Path.home()))).expanduser()
    home_ansys = home / ".ansys"
    try:
        home_ansys.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    raw_host = socket.gethostname()
    hosts = []
    for host in (raw_host, raw_host.split(".", 1)[0]):
        if host and host not in hosts:
            hosts.append(host)
    user = getpass.getuser()

    for host in hosts:
        acl = f"{host}_{host}_{user}_{version}"
        filename = f".ansyscl.{host}.{acl}"
        home_file = home_ansys / filename
        existing_port = _read_ansys_port_file(home_file) if home_file.exists() else None
        if existing_port:
            if home_file.is_symlink():
                port, pid = existing_port
                return f"using existing {home_file} (port {port}, pid {pid})"
            return None
        if home_file.is_symlink():
            try:
                home_file.unlink()
            except OSError:
                return None
        elif home_file.exists():
            return None

        candidates = []
        for candidate in Path("/tmp").glob(f"tfln_lumerical_runtime_*/.ansys/{filename}"):
            if _read_ansys_port_file(candidate):
                candidates.append(candidate)
        if not candidates:
            continue

        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        target = candidates[0]
        try:
            home_file.symlink_to(target)
        except OSError:
            return None
        port, pid = _read_ansys_port_file(target) or (0, 0)
        return f"linked {home_file} -> {target} (port {port}, pid {pid})"

    return None


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
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Run even if the selected GPU already has compute jobs.",
    )
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

    active_gpu_jobs = _active_gpu_compute_jobs(args.gpu)
    if active_gpu_jobs and not args.allow_busy_gpu:
        print(f"Selected GPU {args.gpu} already has compute jobs:", file=sys.stderr)
        for job in active_gpu_jobs:
            print(f"  - {job}", file=sys.stderr)
        print("Use `resource` to pick an allocatable GPU, or pass --allow-busy-gpu to override.", file=sys.stderr)
        return 2

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

    cpu_affinity_warning = None
    allocatable_threads = None
    if not args.no_taskset:
        cpu_affinity, cpu_affinity_warning, allocatable_threads = _choose_cpu_affinity(args.threads)
        env["TASKSET_CPUS"] = cpu_affinity
    else:
        cpu_affinity = "(disabled)"

    install = discover_lumerical()
    if install.lumapi_dir:
        env["LUMERICAL_PYTHONPATH"] = env.get("LUMERICAL_PYTHONPATH", str(install.lumapi_dir))
    if install.bin_dir:
        env["LUMERICAL_BIN_DIR"] = env.get("LUMERICAL_BIN_DIR", str(install.bin_dir))
        env["PATH"] = f"{install.bin_dir}:{env.get('PATH', '')}"
    if install.root:
        env["LUMERICAL_ROOT"] = env.get("LUMERICAL_ROOT", str(install.root))
    ansys_link_message = _ensure_ansys_shared_port_link(env)

    tag = args.tag or f"{script.stem}_gpu{args.gpu}_th{args.threads}"
    desc = args.desc or f"GPU {args.gpu}, FDTD threads {args.threads}"
    cmd = _command(script, cpu_affinity, not args.no_taskset)
    if args.outdir:
        output_base = Path(args.outdir).expanduser()
        output_warning = None
    else:
        output_base, output_warning = _default_output_base()
    try:
        ongoing_root, done_root, failed_root = _prepare_lifecycle_dirs(output_base)
    except OSError as exc:
        print(f"Failed to create Lumerical lifecycle directories under: {output_base}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print("Use --outdir /path/you/can/write.", file=sys.stderr)
        return 2
    run_dir = _run_dir(tag, ongoing_root)
    log_path = run_dir / f"{script.stem}_output.log"
    env["EIDL_RUN_DIR"] = str(run_dir)
    env["EIDL_LUMERICAL_DATA_DIR"] = str(output_base)
    _prepend_pythonpath(env, [launch_cwd, script_dir])

    print("EIDL-Lumapi run")
    print(f"  script                    : {script}")
    print(f"  GPU                       : {args.gpu}")
    if active_gpu_jobs:
        print(f"  GPU warning               : selected GPU has active jobs: {'; '.join(active_gpu_jobs)}")
    print(f"  FDTD threads              : {args.threads}")
    print(f"  CPU affinity              : {env.get('TASKSET_CPUS', '(disabled)')}")
    if allocatable_threads is not None:
        print(f"  allocatable CPU threads   : {allocatable_threads}")
    if cpu_affinity_warning:
        print(f"  CPU affinity warning      : {cpu_affinity_warning}")
    print(f"  LUMERICAL_ROOT            : {env.get('LUMERICAL_ROOT', '(not detected)')}")
    print(f"  LUMERICAL_PYTHONPATH      : {env.get('LUMERICAL_PYTHONPATH', '(not detected)')}")
    print(f"  profiling                 : {'on' if env.get('MSOPT_PROFILE') == '1' else 'off'}")
    if ansys_link_message:
        print(f"  Ansys license sharing     : {ansys_link_message}")
    if output_warning:
        print(f"  output warning            : {output_warning}")
    print(f"  execution cwd             : {run_dir}")
    print(f"  output directory          : {run_dir}")
    print("  lifecycle                 : Ongoing -> Done/Failed")
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
            f"GPU active jobs: {'; '.join(active_gpu_jobs) if active_gpu_jobs else '(none)'}",
            f"FDTD threads: {args.threads}",
            f"CPU affinity: {env.get('TASKSET_CPUS', '(disabled)')}",
            f"Allocatable CPU threads: {allocatable_threads if allocatable_threads is not None else '(disabled)'}",
            f"CPU affinity warning: {cpu_affinity_warning or '(none)'}",
            f"LUMERICAL_ROOT: {env.get('LUMERICAL_ROOT', '(not detected)')}",
            f"LUMERICAL_PYTHONPATH: {env.get('LUMERICAL_PYTHONPATH', '(not detected)')}",
            f"Ansys license sharing: {ansys_link_message or '(default)'}",
            f"Command: {' '.join(cmd)}",
            f"Output log: {log_path}",
            f"Profiling: {'on' if env.get('MSOPT_PROFILE') == '1' else 'off'}",
            f"Profiling output: {run_dir / 'profiling_results'}",
        ],
    )
    final_dir = run_dir
    return_code = 1
    interrupted = False
    try:
        return_code = _stream_process(cmd, env, log_path, run_dir)
    except KeyboardInterrupt:
        interrupted = True
        return_code = 130
        print("\nEIDL-Lumapi run interrupted by user.", file=sys.stderr)
    except BaseException as exc:
        interrupted = True
        return_code = 1
        print(f"\nEIDL-Lumapi run wrapper failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        if run_dir.exists():
            with (run_dir / "info.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"Exit code: {return_code}\n")
                handle.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
                if interrupted:
                    handle.write("Lifecycle status: interrupted/failed\n")
            destination_root = done_root if return_code == 0 else failed_root
            try:
                final_dir = _move_run_dir(run_dir, destination_root)
            except OSError as exc:
                final_dir = run_dir
                print(f"Failed to move run directory to {destination_root}: {exc}", file=sys.stderr)
    log_path = final_dir / f"{script.stem}_output.log"
    print(f"\nEIDL-Lumapi run finished with exit code {return_code}")
    print(f"Result directory: {final_dir}")
    print(f"Log: {log_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
