from __future__ import annotations

import argparse
import os
import pwd
import re
import subprocess
import time
from dataclasses import dataclass

from .common import discover_lumerical, ensure_first_run_config, license_path, run_text

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass(frozen=True)
class LicenseUsage:
    feature: str
    total: int | None
    used: int | None
    source: str

    @property
    def available(self) -> int | None:
        if self.total is None or self.used is None:
            return None
        return max(self.total - self.used, 0)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    mem_total_mib: int
    mem_used_mib: int
    util_pct: int

    @property
    def free_mib(self) -> int:
        return max(self.mem_total_mib - self.mem_used_mib, 0)


@dataclass(frozen=True)
class GpuJob:
    gpu_index: int
    pid: int
    process_name: str
    used_memory_mib: int
    task_name: str
    start_time: float | None = None
    owner: str | None = None

    @property
    def elapsed_seconds(self) -> float | None:
        if self.start_time is None:
            return None
        return max(time.time() - self.start_time, 0.0)


@dataclass
class GpuPeak:
    index: int
    name: str
    mem_total_mib: int
    peak_util_pct: int = 0
    peak_mem_used_mib: int = 0
    peak_util_jobs: list[GpuJob] | None = None
    peak_mem_jobs: list[GpuJob] | None = None
    saw_jobs: bool = False
    current: GpuInfo | None = None
    samples: int = 0

    @property
    def peak_free_mib(self) -> int:
        return max(self.mem_total_mib - self.peak_mem_used_mib, 0)


@dataclass
class LicensePeak:
    feature: str
    total: int | None = None
    peak_used: int | None = None
    current_used: int | None = None
    source: str = ""
    samples: int = 0

    @property
    def available_at_peak(self) -> int | None:
        if self.total is None or self.peak_used is None:
            return None
        return max(self.total - self.peak_used, 0)


def _license_from_lmstat(feature: str, license_server: str | None) -> LicenseUsage:
    install = discover_lumerical()
    if not install.lmutil:
        return LicenseUsage(feature, None, None, "lmutil not found")

    cmd = [str(install.lmutil), "lmstat", "-a"]
    if license_server:
        cmd.extend(["-c", license_server])
    text = run_text(cmd, timeout=12)
    source = f"{install.lmutil}"
    if license_server:
        source += f" -c {license_server}"

    pattern = re.compile(
        rf"Users of {re.escape(feature)}:\s+\(Total of\s+(\d+)\s+licenses issued;\s+Total of\s+(\d+)\s+licenses in use\)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return LicenseUsage(feature, None, None, source)
    return LicenseUsage(feature, int(match.group(1)), int(match.group(2)), source)


def _gpus() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    gpus: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    uuid=parts[1],
                    name=parts[2],
                    mem_total_mib=int(parts[3]),
                    mem_used_mib=int(parts[4]),
                    util_pct=int(parts[5]),
                )
            )
        except ValueError:
            continue
    return gpus


def _proc_cmdline(pid: int) -> list[str]:
    if psutil is not None:
        try:
            return psutil.Process(pid).cmdline()
        except Exception:
            pass
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _proc_environ(pid: int) -> dict[str, str]:
    if psutil is not None:
        try:
            return psutil.Process(pid).environ()
        except Exception:
            pass
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def _proc_start_time(pid: int) -> float | None:
    """Epoch seconds when the process started, or None if it cannot be read."""
    if psutil is not None:
        try:
            return float(psutil.Process(pid).create_time())
        except Exception:
            pass
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat = handle.read().decode(errors="replace")
        with open("/proc/uptime", "rb") as handle:
            uptime = float(handle.read().split()[0])
    except (OSError, IndexError, ValueError):
        return None
    # Field 22 (starttime) is counted after the comm field, which may contain spaces.
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        ticks = float(fields[19])
    except ValueError:
        return None
    clk_tck = os.sysconf("SC_CLK_TCK") or 100
    return (time.time() - uptime) + ticks / clk_tck


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "t = unknown"
    total_minutes = int(seconds // 60)
    return f"t = {total_minutes // 60}h {total_minutes % 60}min"


def _ppid(pid: int) -> int | None:
    """Parent pid straight from /proc, readable even for another user's process."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat = handle.read().decode(errors="replace")
    except OSError:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _parent_pids(pid: int, limit: int = 12) -> list[int]:
    pids = [pid]
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            for _ in range(limit):
                parent = proc.parent()
                if parent is None:
                    break
                pids.append(parent.pid)
                proc = parent
            return pids
        except Exception:
            pids = [pid]
    current = pid
    for _ in range(limit):
        parent = _ppid(current)
        if parent is None or parent <= 1:
            break
        pids.append(parent)
        current = parent
    return pids


SCRIPT_SUFFIXES = (".py", ".sh")


def _script_arg(cmdline: list[str]) -> str | None:
    for arg in cmdline:
        if arg.endswith(SCRIPT_SUFFIXES):
            return arg
    return None


def _command_label(cmdline: list[str]) -> str | None:
    """The command's own name, skipping the interpreter and its option flags."""
    for arg in cmdline:
        if not arg or arg.startswith("-"):
            continue
        base = os.path.basename(arg)
        if not base or "python" in base.lower():
            continue
        return base
    return None


PROJECT_SUFFIXES = (".fsp", ".lms", ".ldev", ".icp")


def _project_arg(cmdline: list[str]) -> str | None:
    for arg in cmdline:
        if arg.endswith(PROJECT_SUFFIXES):
            return arg
    return None


def _abs_path(path: str, pid: int) -> str | None:
    if os.path.isabs(path):
        return path
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")  # readable only for own processes
    except OSError:
        return None
    return os.path.join(cwd, path)


_script_search_cache: dict[tuple[str, str], str | None] = {}

SEARCH_MAX_DEPTH = 5
SEARCH_SKIP_DIRS = frozenset(
    {
        "miniconda3",
        "anaconda3",
        "node_modules",
        "site-packages",
        "__pycache__",
        "venv",
        "envs",
        "build",
        "dist",
        "lost+found",
    }
)


def _search_roots(owner: str | None) -> list[str]:
    """Where a job's code plausibly lives, most specific first.

    A ramdisk copy is what actually runs when one exists, so /dev/shm is
    searched before the source tree it was copied from.  Personal accounts
    are skipped: /home/<user> only ever yields the user name, which the uid
    already gives -- the shared account is the one that hides who is running.
    """
    roots = ["/dev/shm"]
    if owner == SHARED_ACCOUNT:
        roots += [f"/home/{owner}", f"/data/{owner}"]
    return [root for root in roots if os.path.isdir(root)]


def _find_script_path(script: str, owner: str | None) -> str | None:
    """Locate a relative script when its process's cwd is unreadable.

    /proc/<pid>/cwd is owner-only, so another user's job hides where it runs
    from -- but the directories themselves are world-readable, so the script
    name can be searched under the places that account runs from.  Only an
    unambiguous answer is trusted: hits that disagree on the owner folder
    could label the job as the wrong person, so they are all discarded.
    """
    base = os.path.basename(script)
    if not base:
        return None
    key = (owner or "", base)
    if key in _script_search_cache:
        return _script_search_cache[key]
    result = None
    for root in _search_roots(owner):
        hits: dict[str, str] = {}
        base_depth = root.rstrip("/").count(os.sep)
        for current, dirs, files in os.walk(root, onerror=None):
            if current.count(os.sep) - base_depth >= SEARCH_MAX_DEPTH:
                dirs[:] = []
            else:
                dirs[:] = [
                    name
                    for name in dirs
                    if not name.startswith(".") and name not in SEARCH_SKIP_DIRS
                ]
            if base in files:
                path = os.path.join(current, base)
                label = _owner_folder(path)
                if label:
                    hits[label] = path
                    if len(hits) > 1:
                        break
        if len(hits) == 1:
            result = next(iter(hits.values()))
            break
    _script_search_cache[key] = result
    return result


SHARED_ACCOUNT = "eidl"
USER_ROOTS = ("home", "data")


def _owner_folder(path: str | None) -> str | None:
    """Whose folder the task runs from: /home/<name>, /data/<name>, or a /dev/shm dir.

    eidl is the shared account, so under it the folder below is the label.
    """
    if not path or not path.startswith("/"):
        return None
    parts = [part for part in path.split("/") if part]
    if not os.path.isdir(path):
        # A file name says nothing about ownership; the folder holding it does.
        parts = parts[:-1]
    if len(parts) >= 3 and parts[0] == "dev" and parts[1] == "shm":
        return parts[2]
    if len(parts) < 2 or parts[0] not in USER_ROOTS:
        return None
    if parts[1] == SHARED_ACCOUNT and len(parts) >= 3:
        return parts[2]
    return parts[1]


def _pid_user(pid: int) -> str | None:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _task_identity_for_pid(pid: int, process_name: str) -> tuple[str, int, str | None]:
    """Task label, the pid it was resolved from, and the path that identified it."""
    run_dir_owner: tuple[str, int] | None = None
    script_owner: tuple[str, int] | None = None
    candidates = _parent_pids(pid)
    for candidate_pid in candidates:
        env = _proc_environ(candidate_pid)
        run_dir = env.get("EIDL_RUN_DIR")
        if run_dir:
            if run_dir_owner is not None and run_dir_owner[0] != run_dir:
                break
            # EIDL_RUN_DIR is inherited, so keep walking: the oldest ancestor
            # carrying the same run dir is the study runner, not the restarted worker.
            run_dir_owner = (run_dir, candidate_pid)
            continue
        if run_dir_owner is not None:
            break

        # A script name is what identifies the task, so the whole ancestry is
        # searched for one.  The process holding the card is often not a script
        # at all -- fdtd-engine is a binary, restarted for every solve -- and
        # naming it would peg elapsed to a worker minutes old inside a run that
        # has been going for hours.  /proc/<pid>/environ is readable only by the
        # owner, so for anyone else's job this walk is the only thing that works.
        script = _script_arg(_proc_cmdline(candidate_pid))
        if script:
            script_owner = (script, candidate_pid)
            break

    if run_dir_owner is not None:
        run_dir, owner_pid = run_dir_owner
        name = os.path.basename(run_dir.rstrip("/")) or run_dir
        # Run dirs all live in the shared Lumerical_data area; the runner's
        # script path is what says whose folder the study belongs to.
        script = _script_arg(_proc_cmdline(owner_pid))
        label_path = _abs_path(script, owner_pid) if script else None
        return name, owner_pid, label_path or run_dir

    # The engine's command line names the project file sitting inside the run
    # dir, and unlike environ it is readable for anyone's process -- so another
    # user's solve still resolves to its run dir instead of a bare script name.
    project = _project_arg(_proc_cmdline(pid))
    if project:
        project_path = _abs_path(project, pid)
        if project_path:
            run_dir_name = os.path.basename(os.path.dirname(project_path))
            if run_dir_name:
                owner_pid = script_owner[1] if script_owner else pid
                return run_dir_name, owner_pid, project_path

    if script_owner is not None:
        script, owner_pid = script_owner
        path = _abs_path(script, owner_pid) or _find_script_path(script, _pid_user(owner_pid))
        return os.path.basename(script), owner_pid, path

    # No script anywhere above the card: the process itself is all there is, and
    # it is at least a real process with a real start time.
    for candidate_pid in candidates:
        label = _command_label(_proc_cmdline(candidate_pid))
        if label:
            return label, candidate_pid, None
    return process_name or "(unknown)", pid, None


def _gpu_jobs(gpus: list[GpuInfo]) -> dict[int, list[GpuJob]]:
    uuid_to_index = {gpu.uuid: gpu.index for gpu in gpus}
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    jobs: dict[int, list[GpuJob]] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid_text, process_name, used_text = parts[:4]
        try:
            pid = int(pid_text)
            used_memory_mib = int(float(used_text))
        except ValueError:
            continue
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue
        task_name, task_pid, task_path = _task_identity_for_pid(pid, process_name)
        # Elapsed follows the task owner: engine workers restart per iteration, the study does not.
        start_time = _proc_start_time(task_pid)
        if start_time is None and task_pid != pid:
            start_time = _proc_start_time(pid)
        # Whose job: the /home folder the run lives in, else the process owner.
        owner = _owner_folder(task_path) or _pid_user(task_pid) or _pid_user(pid)
        jobs.setdefault(gpu_index, []).append(
            GpuJob(
                gpu_index=gpu_index,
                pid=pid,
                process_name=os.path.basename(process_name) or process_name,
                used_memory_mib=used_memory_mib,
                task_name=task_name,
                start_time=start_time,
                owner=owner,
            )
        )
    return jobs


def _allocatable_gpus(gpus: list[GpuInfo], max_util: int, min_free_mib: int) -> list[GpuInfo]:
    del max_util, min_free_mib
    jobs = _gpu_jobs(gpus)
    return [
        gpu
        for gpu in gpus
        if not jobs.get(gpu.index)
    ]


def _elapsed_width(job_lists: list[list[GpuJob] | None]) -> int:
    """One column width for the whole report, so owner tags line up across GPUs."""
    return max(
        (
            len(_format_elapsed(job.elapsed_seconds))
            for jobs in job_lists
            if jobs
            for job in jobs
        ),
        default=0,
    )


def _job_lines(jobs: list[GpuJob] | None, width: int = 0) -> list[str]:
    if not jobs:
        return ["no compute jobs detected"]
    ordered = sorted(jobs, key=lambda item: item.used_memory_mib, reverse=True)
    elapsed = [_format_elapsed(job.elapsed_seconds) for job in ordered]
    width = max(width, max(len(text) for text in elapsed))
    return [
        (
            f"{text:<{width}}  /  "
            + (f"[{job.owner}] " if job.owner else "")
            + f"{job.task_name} "
            f"(pid={job.pid}, proc={job.process_name}, vram={job.used_memory_mib} MiB)"
        )
        for text, job in zip(elapsed, ordered)
    ]


def _collect_gpu_history(seconds: float, interval: float) -> tuple[list[GpuInfo], dict[int, GpuPeak]]:
    seconds = max(float(seconds), 0.0)
    interval = max(float(interval), 0.2)
    deadline = time.time() + seconds
    peaks: dict[int, GpuPeak] = {}
    last_gpus: list[GpuInfo] = []

    while True:
        gpus = _gpus()
        jobs = _gpu_jobs(gpus)
        last_gpus = gpus
        for gpu in gpus:
            peak = peaks.setdefault(
                gpu.index,
                GpuPeak(
                    index=gpu.index,
                    name=gpu.name,
                    mem_total_mib=gpu.mem_total_mib,
                ),
            )
            peak.current = gpu
            peak.samples += 1
            gpu_jobs = jobs.get(gpu.index, [])
            if gpu_jobs:
                peak.saw_jobs = True
            if gpu.util_pct >= peak.peak_util_pct:
                peak.peak_util_pct = gpu.util_pct
                peak.peak_util_jobs = list(gpu_jobs)
            if gpu.mem_used_mib >= peak.peak_mem_used_mib:
                peak.peak_mem_used_mib = gpu.mem_used_mib
                peak.peak_mem_jobs = list(gpu_jobs)

        if seconds <= 0 or time.time() >= deadline:
            break
        time.sleep(min(interval, max(deadline - time.time(), 0.0)))

    return last_gpus, peaks


def _collect_resource_history(
    seconds: float,
    gpu_interval: float,
    license_interval: float,
    feature: str,
    license_server: str | None,
) -> tuple[list[GpuInfo], dict[int, GpuPeak], LicensePeak]:
    seconds = max(float(seconds), 0.0)
    gpu_interval = max(float(gpu_interval), 0.2)
    license_interval = max(float(license_interval), 1.0)

    deadline = time.time() + seconds
    next_gpu_sample = 0.0
    next_license_sample = 0.0
    last_gpus: list[GpuInfo] = []
    gpu_peaks: dict[int, GpuPeak] = {}
    license_peak = LicensePeak(feature=feature)

    while True:
        now = time.time()

        if now >= next_gpu_sample:
            gpus = _gpus()
            jobs = _gpu_jobs(gpus)
            last_gpus = gpus
            for gpu in gpus:
                peak = gpu_peaks.setdefault(
                    gpu.index,
                    GpuPeak(
                        index=gpu.index,
                        name=gpu.name,
                        mem_total_mib=gpu.mem_total_mib,
                    ),
                )
                peak.current = gpu
                peak.samples += 1
                gpu_jobs = jobs.get(gpu.index, [])
                if gpu_jobs:
                    peak.saw_jobs = True
                if gpu.util_pct >= peak.peak_util_pct:
                    peak.peak_util_pct = gpu.util_pct
                    peak.peak_util_jobs = list(gpu_jobs)
                if gpu.mem_used_mib >= peak.peak_mem_used_mib:
                    peak.peak_mem_used_mib = gpu.mem_used_mib
                    peak.peak_mem_jobs = list(gpu_jobs)
            next_gpu_sample = now + gpu_interval

        if now >= next_license_sample:
            usage = _license_from_lmstat(feature, license_server)
            license_peak.samples += 1
            license_peak.source = usage.source
            if usage.total is not None:
                license_peak.total = usage.total
            if usage.used is not None:
                license_peak.current_used = usage.used
                if license_peak.peak_used is None or usage.used > license_peak.peak_used:
                    license_peak.peak_used = usage.used
            next_license_sample = time.time() + license_interval

        if seconds <= 0 or time.time() >= deadline:
            break

        sleep_until = min(next_gpu_sample, next_license_sample, deadline)
        time.sleep(max(min(sleep_until - time.time(), 0.5), 0.05))

    return last_gpus, gpu_peaks, license_peak


def _allocatable_from_peaks(peaks: dict[int, GpuPeak], max_util: int, min_free_mib: int) -> list[GpuPeak]:
    return [
        peak
        for peak in sorted(peaks.values(), key=lambda item: item.index)
        if not peak.saw_jobs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resource",
        description="Show currently available Lumerical FDTD simulation capacity and GPU allocation candidates.",
    )
    parser.add_argument("--feature", default="lum_fdtd_solve", help="License feature to inspect.")
    parser.add_argument("--license", help="License server/path. Defaults to ANSYSLMD_LICENSE_FILE.")
    parser.add_argument("--max-gpu-util", type=int, default=10, help="Deprecated; allocatable now means no compute jobs detected.")
    parser.add_argument("--min-free-mib", type=int, default=1024, help="Deprecated; allocatable now means no compute jobs detected.")
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=float(os.environ.get("EIDL_RESOURCE_HISTORY_SECONDS", "5")),
        help="Sample resource usage for this many seconds and report peak usage. Default: 5.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("EIDL_RESOURCE_SAMPLE_INTERVAL", "0.5")),
        help="GPU history sample interval in seconds. Default: 0.5.",
    )
    parser.add_argument(
        "--license-sample-interval",
        type=float,
        default=float(os.environ.get("EIDL_RESOURCE_LICENSE_INTERVAL", "10")),
        help="License history sample interval in seconds. Default: 10.",
    )
    parser.add_argument("--instant", action="store_true", help="Skip history sampling and show current usage only.")
    args = parser.parse_args(argv)
    ensure_first_run_config(verbose=True)
    if not args.license:
        args.license = license_path()

    history_seconds = 0.0 if args.instant else max(float(args.history_seconds), 0.0)
    if history_seconds > 0:
        print(
            f"Sampling resources for {history_seconds:g}s "
            f"(gpu_interval={max(float(args.sample_interval), 0.2):g}s, "
            f"license_interval={max(float(args.license_sample_interval), 1.0):g}s)..."
        )
    gpus, gpu_peaks, license_peak = _collect_resource_history(
        history_seconds,
        args.sample_interval,
        args.license_sample_interval,
        args.feature,
        args.license,
    )
    gpu_jobs = _gpu_jobs(gpus)
    if gpu_peaks:
        allocatable_peaks = _allocatable_from_peaks(gpu_peaks, args.max_gpu_util, args.min_free_mib)
        allocatable_gpu_indices = [peak.index for peak in allocatable_peaks]
    else:
        allocatable = _allocatable_gpus(gpus, args.max_gpu_util, args.min_free_mib)
        allocatable_gpu_indices = [gpu.index for gpu in allocatable]

    license_available = license_peak.available_at_peak
    machine_max = len(gpus)
    current_machine_available = len(allocatable_gpu_indices)
    if license_available is None:
        runnable = current_machine_available
        limit_label = "license unknown, machine only"
    else:
        runnable = min(license_available, current_machine_available)
        limit_label = "min(available license, allocatable GPUs)"

    total_limit = machine_max if license_peak.total is None else min(license_peak.total, machine_max)

    print("EIDL-Lumapi resources")
    print(f"  Lumerical feature       : {args.feature}")
    print(f"  License source          : {license_peak.source or '(unknown)'}")
    print(f"  License server/path     : {args.license or '(not set)'}")
    if license_peak.total is None or license_peak.peak_used is None:
        print("  License usage           : unknown")
    else:
        print(
            f"  License usage peak      : {license_peak.peak_used}/{license_peak.total} in use, "
            f"{license_peak.available_at_peak} available "
            f"(samples={license_peak.samples})"
        )
        if license_peak.current_used is not None:
            print(f"  License usage current   : {license_peak.current_used}/{license_peak.total} in use")
    print(f"  Machine GPUs            : {machine_max}")
    print(f"  History window          : {history_seconds:g}s")
    print(f"  Allocatable GPUs        : {', '.join(str(index) for index in allocatable_gpu_indices) or '(none)'}")
    print(f"  Runnable simulations    : {runnable}/{total_limit} ({limit_label})")

    if gpus:
        print("")
        print("GPU status")
        ordered_gpus = sorted(gpus, key=lambda item: item.index)
        shown_jobs = {
            gpu.index: (
                gpu_peaks[gpu.index].peak_util_jobs
                if gpu.index in gpu_peaks
                else gpu_jobs.get(gpu.index, [])
            )
            for gpu in ordered_gpus
        }
        elapsed_width = _elapsed_width(list(shown_jobs.values()))
        for position, gpu in enumerate(ordered_gpus):
            if position:
                print("")
            marker = "*" if gpu.index in allocatable_gpu_indices else " "
            peak = gpu_peaks.get(gpu.index)
            print(
                f" {marker} GPU {gpu.index}: {gpu.name}, "
                f"util={gpu.util_pct}%, mem={gpu.mem_used_mib}/{gpu.mem_total_mib} MiB, "
                f"free={gpu.free_mib} MiB"
            )
            if peak:
                print(
                    f"     peak over {history_seconds:g}s: "
                    f"util={peak.peak_util_pct}%, "
                    f"mem={peak.peak_mem_used_mib}/{peak.mem_total_mib} MiB, "
                    f"free={peak.peak_free_mib} MiB, samples={peak.samples}"
                )
                print("     jobs at peak util:")
                for line in _job_lines(shown_jobs[gpu.index], elapsed_width):
                    print(f"       - {line}")
            else:
                print("     peak over history: unavailable")
                print("     current jobs:")
                for line in _job_lines(shown_jobs[gpu.index], elapsed_width):
                    print(f"       - {line}")
        print("")
        print("  * = no compute jobs detected during the history window")
    else:
        print("")
        print("GPU status")
        print("  nvidia-smi not found or no GPUs detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
