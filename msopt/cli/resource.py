from __future__ import annotations

import argparse
import os
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


def _parent_pids(pid: int, limit: int = 8) -> list[int]:
    pids = [pid]
    if psutil is None:
        return pids
    try:
        proc = psutil.Process(pid)
        for _ in range(limit):
            parent = proc.parent()
            if parent is None:
                break
            pids.append(parent.pid)
            proc = parent
    except Exception:
        pass
    return pids


def _task_name_for_pid(pid: int, process_name: str) -> str:
    for candidate_pid in _parent_pids(pid):
        env = _proc_environ(candidate_pid)
        run_dir = env.get("EIDL_RUN_DIR")
        if run_dir:
            return os.path.basename(run_dir.rstrip("/")) or run_dir

        cmdline = _proc_cmdline(candidate_pid)
        for arg in cmdline:
            if arg.endswith(".py"):
                return os.path.basename(arg)
        for arg in cmdline:
            if "python" not in os.path.basename(arg).lower() and arg:
                base = os.path.basename(arg)
                if base:
                    return base
    return process_name or "(unknown)"


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
        jobs.setdefault(gpu_index, []).append(
            GpuJob(
                gpu_index=gpu_index,
                pid=pid,
                process_name=os.path.basename(process_name) or process_name,
                used_memory_mib=used_memory_mib,
                task_name=_task_name_for_pid(pid, process_name),
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


def _job_lines(jobs: list[GpuJob] | None) -> list[str]:
    if not jobs:
        return ["no compute jobs detected"]
    return [
        (
            f"{job.task_name} "
            f"(pid={job.pid}, proc={job.process_name}, vram={job.used_memory_mib} MiB)"
        )
        for job in sorted(jobs, key=lambda item: item.used_memory_mib, reverse=True)
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
        for gpu in sorted(gpus, key=lambda item: item.index):
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
                for line in _job_lines(peak.peak_util_jobs):
                    print(f"       - {line}")
            else:
                print("     peak over history: unavailable")
                print("     current jobs:")
                for line in _job_lines(gpu_jobs.get(gpu.index, [])):
                    print(f"       - {line}")
        print("  * = no compute jobs detected during the history window")
    else:
        print("")
        print("GPU status")
        print("  nvidia-smi not found or no GPUs detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
