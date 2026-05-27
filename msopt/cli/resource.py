from __future__ import annotations

import argparse
import os
import re
import subprocess
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
    return [
        gpu
        for gpu in gpus
        if gpu.util_pct <= max_util and gpu.free_mib >= min_free_mib
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resource",
        description="Show currently available Lumerical FDTD simulation capacity and GPU allocation candidates.",
    )
    parser.add_argument("--feature", default="lum_fdtd_solve", help="License feature to inspect.")
    parser.add_argument("--license", help="License server/path. Defaults to ANSYSLMD_LICENSE_FILE.")
    parser.add_argument("--max-gpu-util", type=int, default=10, help="GPU is allocatable when utilization is <= this value.")
    parser.add_argument("--min-free-mib", type=int, default=1024, help="GPU is allocatable when free VRAM is >= this value.")
    args = parser.parse_args(argv)
    ensure_first_run_config(verbose=True)
    if not args.license:
        args.license = license_path()

    usage = _license_from_lmstat(args.feature, args.license)
    gpus = _gpus()
    gpu_jobs = _gpu_jobs(gpus)
    allocatable = _allocatable_gpus(gpus, args.max_gpu_util, args.min_free_mib)

    license_available = usage.available
    machine_max = len(gpus)
    current_machine_available = len(allocatable)
    if license_available is None:
        runnable = current_machine_available
        limit_label = "license unknown, machine only"
    else:
        runnable = min(license_available, current_machine_available)
        limit_label = "min(available license, allocatable GPUs)"

    total_limit = machine_max if usage.total is None else min(usage.total, machine_max)

    print("EIDL-Lumapi resources")
    print(f"  Lumerical feature       : {args.feature}")
    print(f"  License source          : {usage.source}")
    print(f"  License server/path     : {args.license or '(not set)'}")
    if usage.total is None or usage.used is None:
        print("  License usage           : unknown")
    else:
        print(f"  License usage           : {usage.used}/{usage.total} in use, {usage.available} available")
    print(f"  Machine GPUs            : {machine_max}")
    print(f"  Allocatable GPUs        : {', '.join(str(g.index) for g in allocatable) or '(none)'}")
    print(f"  Runnable simulations    : {runnable}/{total_limit} ({limit_label})")

    if gpus:
        print("")
        print("GPU status")
        for gpu in gpus:
            marker = "*" if gpu in allocatable else " "
            print(
                f" {marker} GPU {gpu.index}: {gpu.name}, "
                f"util={gpu.util_pct}%, mem={gpu.mem_used_mib}/{gpu.mem_total_mib} MiB, "
                f"free={gpu.free_mib} MiB"
            )
            jobs = gpu_jobs.get(gpu.index, [])
            if jobs:
                for job in jobs:
                    print(
                        f"     - {job.task_name} "
                        f"(pid={job.pid}, proc={job.process_name}, vram={job.used_memory_mib} MiB)"
                    )
            else:
                print("     - no compute jobs detected")
        print("  * = allocatable by current thresholds")
    else:
        print("")
        print("GPU status")
        print("  nvidia-smi not found or no GPUs detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
