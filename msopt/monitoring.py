"""Runtime profiling helpers for EIDL-Lumapi simulations."""

from __future__ import annotations

import atexit
import csv
import functools
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is part of the conda env.
    psutil = None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _run_dir() -> Path:
    return Path(os.environ.get("EIDL_RUN_DIR") or os.environ.get("PROFILING_RUN_DIR") or ".").resolve()


def _profile_dir(run_dir: Path | None = None) -> Path:
    return (run_dir or _run_dir()) / "profiling_results"


def _process_tree_pids() -> list[int]:
    pids = [os.getpid()]
    if psutil is None:
        return pids
    try:
        proc = psutil.Process(os.getpid())
        pids.extend(child.pid for child in proc.children(recursive=True))
    except Exception:
        pass
    return pids


def _children_usage() -> dict[str, Any]:
    if psutil is None:
        return {"children_rss_MB": 0.0, "children_threads": 0, "children_pids": []}
    rss = 0
    threads = 0
    pids = []
    try:
        proc = psutil.Process(os.getpid())
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
                threads += child.num_threads()
                pids.append(child.pid)
            except Exception:
                pass
    except Exception:
        pass
    return {
        "children_rss_MB": rss / (1024 * 1024),
        "children_threads": threads,
        "children_pids": pids,
    }


def _gpu_usage() -> dict[str, Any]:
    overview: list[dict[str, Any]] = []
    process_vram: dict[str, int] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            overview.append(
                {
                    "index": int(parts[0]),
                    "mem_used_MiB": int(float(parts[1])),
                    "mem_total_MiB": int(float(parts[2])),
                    "util_pct": int(float(parts[3])),
                }
            )
    except Exception:
        pass

    pids = set(_process_tree_pids())
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[1])
                used = int(float(parts[2]))
            except ValueError:
                continue
            if pid in pids:
                process_vram[parts[0]] = process_vram.get(parts[0], 0) + used
    except Exception:
        pass

    return {"overview": overview, "process_vram_MiB": process_vram}


def _sample() -> dict[str, Any]:
    now = time.perf_counter()
    if psutil is None:
        return {"t": now, "rss_MB": 0.0, "threads": 0, **_gpu_usage()}
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    children = _children_usage()
    return {
        "t": now,
        "rss_MB": mem.rss / (1024 * 1024),
        "vms_MB": mem.vms / (1024 * 1024),
        "threads": proc.num_threads(),
        "cpu_percent": proc.cpu_percent(interval=None),
        "children_rss_MB": children["children_rss_MB"],
        "children_threads": children["children_threads"],
        "children_pids": children["children_pids"],
        "total_rss_MB": mem.rss / (1024 * 1024) + children["children_rss_MB"],
        "total_threads": proc.num_threads() + children["children_threads"],
        **_gpu_usage(),
    }


def _short_list(items: list[str], limit: int = 30) -> str:
    if not items:
        return ""
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f", ... (+{len(items) - limit} more)"
    return ", ".join(shown) + suffix


def _rough_size_bytes(value: Any, depth: int = 0, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    try:
        size = sys.getsizeof(value)
    except Exception:
        size = 0
    if depth >= 2:
        return size
    if isinstance(value, dict):
        for key, item in list(value.items())[:200]:
            size += _rough_size_bytes(key, depth + 1, seen)
            size += _rough_size_bytes(item, depth + 1, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in list(value)[:200]:
            size += _rough_size_bytes(item, depth + 1, seen)
    elif hasattr(value, "nbytes"):
        try:
            size = max(size, int(value.nbytes))
        except Exception:
            pass
    return size


def _variable_snapshot(locals_dict: dict | None, globals_dict: dict | None) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for source in (globals_dict or {}, locals_dict or {}):
        for name, value in source.items():
            if name.startswith("__"):
                continue
            if getattr(value, "__module__", None) == "builtins" and callable(value):
                continue
            namespace[name] = value
    names = sorted(namespace)
    total_size = 0
    for value in namespace.values():
        total_size += _rough_size_bytes(value)
    return {
        "names": names,
        "count": len(names),
        "size_MB": total_size / (1024 * 1024),
    }


class _Sampler(threading.Thread):
    def __init__(self, interval: float):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.samples.append(_sample())
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)


@dataclass
class StepRecord:
    name: str
    call_index: int
    start_time: str
    elapsed_sec: float
    start: dict[str, Any]
    end: dict[str, Any]
    samples: list[dict[str, Any]] = field(default_factory=list)
    variables_start: dict[str, Any] = field(default_factory=dict)
    variables_end: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        peaks = _peaks(self.samples or [self.start, self.end])
        start_names = set(self.variables_start.get("names", []))
        end_names = set(self.variables_end.get("names", []))
        new_names = sorted(end_names - start_names)
        removed_names = sorted(start_names - end_names)
        return {
            "name": self.name,
            "call_index": self.call_index,
            "start_time": self.start_time,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "python_rss_start_MB": round(float(self.start.get("rss_MB", 0.0)), 2),
            "python_rss_end_MB": round(float(self.end.get("rss_MB", 0.0)), 2),
            "python_rss_peak_MB": peaks["python_rss_peak_MB"],
            "children_rss_peak_MB": peaks["children_rss_peak_MB"],
            "total_rss_peak_MB": peaks["total_rss_peak_MB"],
            "python_threads_peak": peaks["python_threads_peak"],
            "children_threads_peak": peaks["children_threads_peak"],
            "total_threads_peak": peaks["total_threads_peak"],
            "cpu_percent_peak": peaks["cpu_percent_peak"],
            "active_logical_cores_peak": peaks["active_logical_cores_peak"],
            "gpu_global_vram_peak_MiB": peaks["gpu_global_vram_peak_MiB"],
            "gpu_util_peak_pct": peaks["gpu_util_peak_pct"],
            "process_gpu_vram_peak_MiB": peaks["process_gpu_vram_peak_MiB"],
            "variables_count": self.variables_end.get("count", self.variables_start.get("count", 0)),
            "variables_size_MB": round(
                float(self.variables_end.get("size_MB", self.variables_start.get("size_MB", 0.0))),
                4,
            ),
            "variables_present": sorted(end_names or start_names),
            "new_variables": new_names,
            "removed_variables": removed_names,
        }


def _peaks(samples: list[dict[str, Any]]) -> dict[str, Any]:
    gpu_mem: dict[str, int] = {}
    gpu_util: dict[str, int] = {}
    proc_gpu: dict[str, int] = {}
    for sample in samples:
        for gpu in sample.get("overview", []):
            idx = str(gpu.get("index"))
            gpu_mem[idx] = max(gpu_mem.get(idx, 0), int(gpu.get("mem_used_MiB", 0)))
            gpu_util[idx] = max(gpu_util.get(idx, 0), int(gpu.get("util_pct", 0)))
        for key, value in sample.get("process_vram_MiB", {}).items():
            proc_gpu[str(key)] = max(proc_gpu.get(str(key), 0), int(value))

    return {
        "python_rss_peak_MB": round(max(float(s.get("rss_MB", 0.0)) for s in samples), 2),
        "children_rss_peak_MB": round(max(float(s.get("children_rss_MB", 0.0)) for s in samples), 2),
        "total_rss_peak_MB": round(max(float(s.get("total_rss_MB", 0.0)) for s in samples), 2),
        "python_threads_peak": max(int(s.get("threads", 0)) for s in samples),
        "children_threads_peak": max(int(s.get("children_threads", 0)) for s in samples),
        "total_threads_peak": max(int(s.get("total_threads", 0)) for s in samples),
        "cpu_percent_peak": round(max(float(s.get("cpu_percent", 0.0)) for s in samples), 2),
        "active_logical_cores_peak": round(max(float(s.get("cpu_percent", 0.0)) for s in samples) / 100.0, 2),
        "gpu_global_vram_peak_MiB": gpu_mem,
        "gpu_util_peak_pct": gpu_util,
        "process_gpu_vram_peak_MiB": proc_gpu,
    }


def _gb(value_mb: float | int | None) -> float | str:
    if value_mb is None:
        return ""
    return round(float(value_mb) / 1024.0, 3)


def _max_gpu0_vram_gb(records: list[dict[str, Any]]) -> float:
    values = []
    for record in records:
        gpu_mem = record.get("gpu_global_vram_peak_MiB", {})
        if isinstance(gpu_mem, dict) and "0" in gpu_mem:
            values.append(float(gpu_mem["0"]))
    return float(max(values)) / 1024.0 if values else 0.0


def _max_gpu0_util(records: list[dict[str, Any]]) -> float:
    values = []
    for record in records:
        gpu_util = record.get("gpu_util_peak_pct", {})
        if isinstance(gpu_util, dict) and "0" in gpu_util:
            values.append(float(gpu_util["0"]))
    return max(values) if values else 0.0


def _leak_risk(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no"
    first = records[0]
    last = records[-1]
    start = float(first.get("python_rss_start_MB", 0.0))
    end = float(last.get("python_rss_end_MB", 0.0))
    return "yes" if end - start > 50.0 and end > start * 1.15 else "no"


def _group_step_dicts(records: list[StepRecord]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.name].append(record.to_dict())
    return grouped


class ResourceProfiler:
    def __init__(self, sample_interval: float = 0.5, output_dir: str | Path | None = None):
        self.sample_interval = float(sample_interval)
        self.output_dir = Path(output_dir) if output_dir else _profile_dir()
        self.records: list[StepRecord] = []
        self._call_counts: dict[str, int] = defaultdict(int)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_log = self.output_dir / "step_records.jsonl"
        self.step_log.write_text("")

    @contextmanager
    def step(self, name: str, locals_dict: dict | None = None, globals_dict: dict | None = None):
        self._call_counts[name] += 1
        call_index = self._call_counts[name]
        start_wall = datetime.now().isoformat(timespec="seconds")
        variables_start = _variable_snapshot(locals_dict, globals_dict)
        start = _sample()
        sampler = _Sampler(self.sample_interval)
        sampler.start()
        start_t = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_t
            sampler.stop()
            end = _sample()
            variables_end = _variable_snapshot(locals_dict, globals_dict)
            record = StepRecord(
                name=name,
                call_index=call_index,
                start_time=start_wall,
                elapsed_sec=elapsed,
                start=start,
                end=end,
                samples=sampler.samples,
                variables_start=variables_start,
                variables_end=variables_end,
            )
            self.records.append(record)
            with self.step_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), default=str) + "\n")

    def save(self, path: str | Path | None = None, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path) if path else self.output_dir / "profile_result.json"
        metadata = metadata or {}
        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "metadata": metadata,
            "total_steps": len(self.records),
            "steps": [record.to_dict() for record in self.records],
            "summary": self.summary(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        self.export_ppt_csvs(metadata=metadata)
        print(f"[Profiler] saved: {output}")
        return output

    def summary(self) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            grouped[record.name].append(record.to_dict())

        out: dict[str, Any] = {}
        for name, records in grouped.items():
            times = [float(r["elapsed_sec"]) for r in records]
            out[name] = {
                "call_count": len(records),
                "time_total_sec": round(sum(times), 4),
                "time_avg_sec": round(sum(times) / len(times), 4),
                "time_max_sec": round(max(times), 4),
                "total_rss_peak_max_MB": max(float(r["total_rss_peak_MB"]) for r in records),
                "children_rss_peak_max_MB": max(float(r["children_rss_peak_MB"]) for r in records),
                "total_threads_peak_max": max(int(r["total_threads_peak"]) for r in records),
            }
        return out

    def print_summary(self) -> None:
        print("\n" + "=" * 100)
        print("RESOURCE PROFILING SUMMARY")
        print("=" * 100)
        print(f"{'Step':<38} {'Calls':>6} {'Total(s)':>10} {'Avg(s)':>10} {'RSS(MB)':>10} {'Threads':>8}")
        print("-" * 100)
        for name, item in self.summary().items():
            print(
                f"{name:<38} {item['call_count']:>6} "
                f"{item['time_total_sec']:>10.2f} {item['time_avg_sec']:>10.2f} "
                f"{item['total_rss_peak_max_MB']:>10.1f} {item['total_threads_peak_max']:>8}"
            )

    def export_ppt_csvs(self, metadata: dict[str, Any] | None = None) -> list[Path]:
        metadata = metadata or {}
        fdtd_threads = metadata.get("fdtd_threads") or os.environ.get("FDTD_THREADS") or ""
        run_folder = Path(str(metadata.get("run_dir") or _run_dir())).name
        description = metadata.get("description") or "EIDL-Lumapi profiling export"
        grouped = _group_step_dicts(self.records)
        step_rows: list[dict[str, Any]] = []
        variable_rows: list[dict[str, Any]] = []

        for stage, records in grouped.items():
            times = [float(r["elapsed_sec"]) for r in records]
            start_mb = float(records[0].get("python_rss_start_MB", 0.0))
            end_mb = float(records[-1].get("python_rss_end_MB", 0.0))
            present = sorted({name for r in records for name in r.get("variables_present", [])})
            new_vars = sorted({name for r in records for name in r.get("new_variables", [])})
            removed_vars = sorted({name for r in records for name in r.get("removed_variables", [])})
            tracked = max(int(r.get("variables_count", 0)) for r in records)
            var_size_mb = max(float(r.get("variables_size_MB", 0.0)) for r in records)
            step_rows.append(
                {
                    "Stage": stage,
                    "FDTD Threads": fdtd_threads,
                    "Calls": len(records),
                    "Total Time (s)": round(sum(times), 4),
                    "Avg Time (s)": round(sum(times) / len(times), 4),
                    "Max Time (s)": round(max(times), 4),
                    "Peak Python RAM (GB)": _gb(max(float(r.get("python_rss_peak_MB", 0.0)) for r in records)),
                    "Peak FDTD / Child RAM (GB)": _gb(max(float(r.get("children_rss_peak_MB", 0.0)) for r in records)),
                    "Peak Total RAM (GB)": _gb(max(float(r.get("total_rss_peak_MB", 0.0)) for r in records)),
                    "Net Python RAM Change (GB)": _gb(end_mb - start_mb),
                    "Peak GPU0 VRAM (GB)": round(_max_gpu0_vram_gb(records), 3),
                    "Peak GPU0 Utilization (%)": round(_max_gpu0_util(records), 3),
                    "Peak Total Threads": max(int(r.get("total_threads_peak", 0)) for r in records),
                    "Peak Logical Cores Seen": max(int(r.get("total_threads_peak", 0)) for r in records),
                    "Peak Active Logical Cores": max(float(r.get("active_logical_cores_peak", 0.0)) for r in records),
                    "Peak Physical Cores Seen": round(max(int(r.get("total_threads_peak", 0)) for r in records) / 2.0, 1),
                    "Peak Active Physical Cores": round(max(float(r.get("active_logical_cores_peak", 0.0)) for r in records) / 2.0, 1),
                    "Tracked Variables (#)": tracked,
                    "Variables Snapshot Size (GB)": round(var_size_mb / 1024.0, 3),
                    "New Variables (Unique #)": len(new_vars),
                    "Removed Variables (Unique #)": len(removed_vars),
                    "Swap Peak (kB)": 0.0,
                    "Memory Leak Risk": _leak_risk(records),
                }
            )
            variable_rows.append(
                {
                    "Stage": stage,
                    "FDTD Threads": fdtd_threads,
                    "Calls": len(records),
                    "Tracked Variables (#)": tracked,
                    "Variables Snapshot Size (MB)": round(var_size_mb, 2),
                    "Variables Present": _short_list(present),
                    "New Variables": _short_list(new_vars),
                    "Removed Variables": _short_list(removed_vars),
                }
            )

        def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return path

        outputs = [
            write_csv(
                self.output_dir / "ppt_step_resource_table_aggregated_gb.csv",
                step_rows,
                [
                    "Stage",
                    "FDTD Threads",
                    "Calls",
                    "Total Time (s)",
                    "Avg Time (s)",
                    "Max Time (s)",
                    "Peak Python RAM (GB)",
                    "Peak FDTD / Child RAM (GB)",
                    "Peak Total RAM (GB)",
                    "Net Python RAM Change (GB)",
                    "Peak GPU0 VRAM (GB)",
                    "Peak GPU0 Utilization (%)",
                    "Peak Total Threads",
                    "Peak Logical Cores Seen",
                    "Peak Active Logical Cores",
                    "Peak Physical Cores Seen",
                    "Peak Active Physical Cores",
                    "Tracked Variables (#)",
                    "Variables Snapshot Size (GB)",
                    "New Variables (Unique #)",
                    "Removed Variables (Unique #)",
                    "Swap Peak (kB)",
                    "Memory Leak Risk",
                ],
            ),
            write_csv(
                self.output_dir / "ppt_step_variables_table_aggregated.csv",
                variable_rows,
                [
                    "Stage",
                    "FDTD Threads",
                    "Calls",
                    "Tracked Variables (#)",
                    "Variables Snapshot Size (MB)",
                    "Variables Present",
                    "New Variables",
                    "Removed Variables",
                ],
            ),
        ]

        bottleneck_rows = sorted(step_rows, key=lambda row: float(row["Total Time (s)"]), reverse=True)[:10]
        outputs.append(
            write_csv(
                self.output_dir / "ppt_bottleneck_table_gb.csv",
                [
                    {
                        "Bottleneck Stage": row["Stage"],
                        "FDTD Threads": row["FDTD Threads"],
                        "Calls": row["Calls"],
                        "Total Time (s)": row["Total Time (s)"],
                        "Avg Time per Call (s)": row["Avg Time (s)"],
                        "Max Time per Call (s)": row["Max Time (s)"],
                        "Peak Total RAM (GB)": row["Peak Total RAM (GB)"],
                        "Peak FDTD / Child RAM (GB)": row["Peak FDTD / Child RAM (GB)"],
                        "Peak GPU0 VRAM (GB)": row["Peak GPU0 VRAM (GB)"],
                        "Peak Active Logical Cores": row["Peak Active Logical Cores"],
                        "Memory Leak Risk": row["Memory Leak Risk"],
                    }
                    for row in bottleneck_rows
                ],
                [
                    "Bottleneck Stage",
                    "FDTD Threads",
                    "Calls",
                    "Total Time (s)",
                    "Avg Time per Call (s)",
                    "Max Time per Call (s)",
                    "Peak Total RAM (GB)",
                    "Peak FDTD / Child RAM (GB)",
                    "Peak GPU0 VRAM (GB)",
                    "Peak Active Logical Cores",
                    "Memory Leak Risk",
                ],
            )
        )

        all_records = [record.to_dict() for record in self.records]
        if all_records:
            python_start = float(all_records[0].get("python_rss_start_MB", 0.0))
            python_end = float(all_records[-1].get("python_rss_end_MB", 0.0))
        else:
            python_start = 0.0
            python_end = 0.0
        outputs.append(
            write_csv(
                self.output_dir / "ppt_overall_summary_table_aggregated_gb.csv",
                [
                    {
                        "Run Folder": run_folder,
                        "Description": description,
                        "FDTD Threads": fdtd_threads,
                        "Recorded Step Entries": len(self.records),
                        "Unique Stages": len(grouped),
                        "Python RAM at Start (GB)": _gb(python_start),
                        "Python RAM at Last Record End (GB)": _gb(python_end),
                        "Net Python RAM Change (GB)": _gb(python_end - python_start),
                        "Overall Memory Leak Risk": _leak_risk(all_records),
                        "Max Total RAM Peak (GB)": _gb(max((float(r.get("total_rss_peak_MB", 0.0)) for r in all_records), default=0.0)),
                        "Max FDTD / Child RAM Peak (GB)": _gb(max((float(r.get("children_rss_peak_MB", 0.0)) for r in all_records), default=0.0)),
                        "Max GPU0 VRAM Peak (GB)": round(_max_gpu0_vram_gb(all_records), 3),
                        "Max Logical Cores Seen": max((int(r.get("total_threads_peak", 0)) for r in all_records), default=0),
                        "Max Active Logical Cores": max((float(r.get("active_logical_cores_peak", 0.0)) for r in all_records), default=0.0),
                        "Unique Variables Appeared": len({name for r in all_records for name in r.get("new_variables", [])}),
                        "Unique Variables Removed": len({name for r in all_records for name in r.get("removed_variables", [])}),
                    }
                ],
                [
                    "Run Folder",
                    "Description",
                    "FDTD Threads",
                    "Recorded Step Entries",
                    "Unique Stages",
                    "Python RAM at Start (GB)",
                    "Python RAM at Last Record End (GB)",
                    "Net Python RAM Change (GB)",
                    "Overall Memory Leak Risk",
                    "Max Total RAM Peak (GB)",
                    "Max FDTD / Child RAM Peak (GB)",
                    "Max GPU0 VRAM Peak (GB)",
                    "Max Logical Cores Seen",
                    "Max Active Logical Cores",
                    "Unique Variables Appeared",
                    "Unique Variables Removed",
                ],
            )
        )

        for output in outputs:
            print(f"[Profiler] saved CSV: {output}")
        return outputs


class MonitoringSession:
    def __init__(
        self,
        sample_interval: float = 0.5,
        capture_vars: bool = False,
        run_dir: str | Path | None = None,
        profile_filename: str = "profile_result.json",
    ):
        del capture_vars
        self.run_dir = Path(run_dir).resolve() if run_dir else _run_dir()
        self.output_dir = _profile_dir(self.run_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_output = self.output_dir / profile_filename
        self.metadata: dict[str, Any] = {}
        self.profiler = ResourceProfiler(sample_interval=sample_interval, output_dir=self.output_dir)

    def add_metadata(self, **metadata: Any) -> dict[str, Any]:
        self.metadata.update(metadata)
        return self.metadata

    def step(self, name: str, locals_dict: dict | None = None, globals_dict: dict | None = None):
        return self.profiler.step(name, locals_dict, globals_dict)

    def save(self) -> Path:
        return self.profiler.save(self.profile_output, metadata=self.metadata)

    def print_summary(self) -> None:
        self.profiler.print_summary()


class PatchRegistry:
    def __init__(self):
        self._entries: list[tuple[Any, str, Any]] = []

    def patch(self, owner: Any, attribute: str, replacement: Any) -> Any:
        original = getattr(owner, attribute)
        self._entries.append((owner, attribute, original))
        setattr(owner, attribute, replacement)
        return original

    def restore(self) -> None:
        for owner, attribute, original in reversed(self._entries):
            setattr(owner, attribute, original)
        self._entries.clear()


def _method_globals(method: Any) -> dict[str, Any]:
    return getattr(method, "__globals__", {})


def install_default_lumerical_patches(session: MonitoringSession, lumerical_utill_module: Any, opt_ms_module: Any) -> PatchRegistry:
    profiler = session.profiler
    patches = PatchRegistry()

    simulator_cls = lumerical_utill_module.LumericalFDTDSimulator
    opt_problem_cls = lumerical_utill_module.LumericalOptimizationProblem
    opt_ms_cls = opt_ms_module.OPT_Ms

    original_run = simulator_cls.run

    @functools.wraps(original_run)
    def profiled_run(self, name: str = "fdtd_run", **kwargs: Any):
        if "Adjoint" in name:
            label = "sim.run(Adjoint)"
        elif "Forward" in name:
            label = "sim.run(Forward)"
        else:
            label = f"sim.run({name})"
        with profiler.step(label, vars(self), _method_globals(original_run)):
            return original_run(self, name=name, **kwargs)

    patches.patch(simulator_cls, "run", profiled_run)

    for cls, method_name, label in (
        (opt_problem_cls, "__call__", "opt.__call__"),
        (opt_problem_cls, "forward_run", "opt.forward_run"),
        (opt_problem_cls, "adjoint_dipole_run", "opt.adjoint_dipole_run"),
        (opt_problem_cls, "calculate_gradient", "opt.calculate_gradient"),
        (opt_ms_cls, "Design_update", "OPT_Ms.Design_update"),
        (opt_ms_cls, "Inner_iter", "OPT_Ms.Inner_iter"),
    ):
        original = getattr(cls, method_name)

        @functools.wraps(original)
        def wrapper(self, *args: Any, __original=original, __label=label, **kwargs: Any):
            with profiler.step(__label, vars(self), _method_globals(__original)):
                return __original(self, *args, **kwargs)

        patches.patch(cls, method_name, wrapper)

    return patches


_ACTIVE_SESSION: MonitoringSession | None = None
_ACTIVE_PATCHES: PatchRegistry | None = None
_FINALIZED = False


def activate(
    *,
    sample_interval: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> MonitoringSession:
    global _ACTIVE_SESSION, _ACTIVE_PATCHES
    if _ACTIVE_SESSION is not None:
        return _ACTIVE_SESSION

    from . import Lumerical_utill, Opt_MS2

    interval = sample_interval
    if interval is None:
        interval = float(os.environ.get("MSOPT_PROFILE_INTERVAL", "0.5"))

    session = MonitoringSession(sample_interval=interval)
    session.add_metadata(
        script=os.environ.get("EIDL_SCRIPT_DIR", ""),
        run_dir=str(_run_dir()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        fdtd_threads=os.environ.get("FDTD_THREADS", ""),
    )
    if metadata:
        session.add_metadata(**metadata)

    patches = install_default_lumerical_patches(session, Lumerical_utill, Opt_MS2)
    _ACTIVE_SESSION = session
    _ACTIVE_PATCHES = patches
    atexit.register(finalize)
    print(f"[Profiler] active: {session.output_dir}")
    return session


def activate_from_env() -> MonitoringSession | None:
    if not (_truthy(os.environ.get("MSOPT_PROFILE")) or _truthy(os.environ.get("EIDL_PROFILE"))):
        return None
    return activate()


def finalize() -> None:
    global _FINALIZED
    if _FINALIZED:
        return
    _FINALIZED = True
    if _ACTIVE_SESSION is None:
        return
    try:
        _ACTIVE_SESSION.save()
        _ACTIVE_SESSION.print_summary()
    finally:
        if _ACTIVE_PATCHES is not None:
            _ACTIVE_PATCHES.restore()


__all__ = [
    "MonitoringSession",
    "PatchRegistry",
    "ResourceProfiler",
    "activate",
    "activate_from_env",
    "finalize",
    "install_default_lumerical_patches",
]
