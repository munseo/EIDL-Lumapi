from __future__ import annotations

import os
import re
import shutil
import subprocess
import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LumericalInstall:
    root: Path | None
    lmutil: Path | None
    lumapi_dir: Path | None
    bin_dir: Path | None


CONFIG_NAME = ".eidl_lumapi_config.json"


def config_path() -> Path:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix) / CONFIG_NAME
    return Path.home() / ".config" / "EIDL-Lumapi" / "config.json"


def load_config() -> dict[str, str]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def apply_config_env(config: dict[str, str]) -> None:
    for key in ("LUMERICAL_ROOT", "LUMERICAL_PYTHONPATH", "LUMERICAL_BIN_DIR", "ANSYSLMD_LICENSE_FILE"):
        if config.get(key) and not os.environ.get(key):
            os.environ[key] = config[key]
    if config.get("LUMERICAL_BIN_DIR"):
        bin_dir = config["LUMERICAL_BIN_DIR"]
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(os.pathsep):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}"


def discover_lumerical() -> LumericalInstall:
    apply_config_env(load_config())
    roots: list[Path] = []
    env_root = os.environ.get("LUMERICAL_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lumerical = Path(conda_prefix) / "lumerical"
        if conda_lumerical.exists():
            roots.extend(sorted((path for path in conda_lumerical.glob("v*") if path.is_dir()), reverse=True))
    opt_root = Path("/opt/lumerical")
    if opt_root.exists():
        roots.extend(sorted((path for path in opt_root.glob("v*") if path.is_dir()), reverse=True))

    for root in roots:
        lmutil = root / "licensingclient" / "linx64" / "lmutil"
        lumapi = root / "api" / "python" / "lumapi.py"
        bin_dir = root / "bin"
        if lmutil.exists() or lumapi.exists() or bin_dir.exists():
            return LumericalInstall(
                root=root,
                lmutil=lmutil if lmutil.exists() else None,
                lumapi_dir=lumapi.parent if lumapi.exists() else None,
                bin_dir=bin_dir if bin_dir.exists() else None,
            )

    lmutil_path = shutil.which("lmutil")
    return LumericalInstall(
        root=None,
        lmutil=Path(lmutil_path) if lmutil_path else None,
        lumapi_dir=None,
        bin_dir=None,
    )


def license_path() -> str | None:
    apply_config_env(load_config())
    return (
        os.environ.get("ANSYSLMD_LICENSE_FILE")
        or os.environ.get("LUMERICAL_LICENSE")
        or os.environ.get("LM_LICENSE_FILE")
    )


def run_text(cmd: list[str], timeout: int = 8, env: dict[str, str] | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def parse_int(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _candidate_license_files() -> list[Path]:
    candidates: list[Path] = []
    for value in (
        os.environ.get("ANSYSLMD_LICENSE_FILE"),
        os.environ.get("LUMERICAL_LICENSE"),
        os.environ.get("LM_LICENSE_FILE"),
    ):
        if value and "@" not in value:
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            Path.home() / ".config" / "Lumerical" / "License.ini",
            Path.home() / "ansys_license" / "license.lic",
            Path.home() / "ansys_license" / "License.ini",
        ]
    )
    for base in (Path.home(), Path("/opt/lumerical")):
        if base.exists():
            candidates.extend(base.glob("**/License.ini"))
            candidates.extend(base.glob("**/license.lic"))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            unique.append(resolved)
    return unique


def _license_from_file(path: Path) -> str | None:
    text = path.read_text(errors="ignore")
    host_match = re.search(r"^\s*server\s*=\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if host_match:
        return host_match.group(1).strip()

    server_match = re.search(r"^\s*SERVER\s+\S+\s+\S+\s+(\d+)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if server_match:
        host = re.search(r"^\s*SERVER\s+(\S+)", text, re.IGNORECASE | re.MULTILINE)
        if host:
            return f"{server_match.group(1)}@{host.group(1)}"

    inline = re.search(r"(\d{3,5}@[A-Za-z0-9_.-]+)", text)
    if inline:
        return inline.group(1)
    return str(path)


def discover_license() -> tuple[str | None, list[str]]:
    messages: list[str] = []
    current = license_path()
    if current:
        messages.append(f"license from environment: {current}")
        return current, messages

    files = _candidate_license_files()
    if not files:
        messages.append("no License.ini/license.lic file found")
        return None, messages

    for path in files:
        try:
            value = _license_from_file(path)
        except OSError as exc:
            messages.append(f"failed to read license file {path}: {exc}")
            continue
        if value:
            messages.append(f"license from {path}: {value}")
            return value, messages
    messages.append("license files found, but no usable license server/path parsed")
    return None, messages


def _find_lumerical_tree(root: Path) -> Path | None:
    candidates = []
    if root.is_dir():
        candidates.append(root)
        candidates.extend(root.glob("**/v*"))
        candidates.extend(root.glob("**/opt/lumerical/v*"))
    for candidate in sorted(candidates, reverse=True):
        if (
            candidate.is_dir()
            and (candidate / "bin").is_dir()
            and (candidate / "api" / "python" / "lumapi.py").exists()
        ):
            return candidate
    return None


def _copy_lumerical_tree(source: Path) -> tuple[Path | None, str]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None, "CONDA_PREFIX is not set; cannot install Lumerical into the conda environment"
    target_root = Path(conda_prefix) / "lumerical"
    target = target_root / source.name
    if target.exists():
        return target, f"Lumerical already present in conda environment: {target}"
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=True)
    return target, f"Lumerical copied into conda environment: {target}"


def _install_from_installer() -> tuple[Path | None, str]:
    installer = os.environ.get("EIDL_LUMERICAL_INSTALLER")
    if not installer:
        return None, "Lumerical is not installed and EIDL_LUMERICAL_INSTALLER is not set"
    path = Path(installer).expanduser()
    if not path.exists():
        return None, f"EIDL_LUMERICAL_INSTALLER does not exist: {path}"

    if path.is_dir():
        tree = _find_lumerical_tree(path)
        if not tree:
            return None, f"No Lumerical v* install tree found under installer directory: {path}"
        return _copy_lumerical_tree(tree)

    if path.suffix == ".rpm":
        return None, (
            "RPM installer requires system package installation/root permissions. "
            f"Install it manually or provide an extracted install tree via EIDL_LUMERICAL_INSTALLER: {path}"
        )

    if path.name.endswith(".tar.gz") or path.suffix in {".tgz", ".gz"}:
        try:
            with tempfile.TemporaryDirectory(prefix="eidl_lumerical_") as temp:
                temp_dir = Path(temp)
                with tarfile.open(path, "r:*") as archive:
                    archive.extractall(temp_dir)
                tree = _find_lumerical_tree(temp_dir)
                if not tree:
                    return None, f"No Lumerical v* install tree found inside archive: {path}"
                return _copy_lumerical_tree(tree)
        except (OSError, tarfile.TarError) as exc:
            return None, f"Failed to extract installer archive {path}: {exc}"

    return None, f"Unsupported Lumerical installer type: {path}"


def ensure_first_run_config(verbose: bool = True) -> dict[str, str]:
    path = config_path()
    if path.exists():
        config = load_config()
        apply_config_env(config)
        return config

    messages: list[str] = []
    install = discover_lumerical()
    if install.root:
        messages.append(f"Lumerical install detected: {install.root}")
    else:
        installed_root, reason = _install_from_installer()
        messages.append(reason)
        if installed_root:
            os.environ["LUMERICAL_ROOT"] = str(installed_root)
            install = discover_lumerical()

    lic, lic_messages = discover_license()
    messages.extend(lic_messages)

    config: dict[str, str] = {
        "configured": "true",
        "lumerical_root": str(install.root) if install.root else "",
        "LUMERICAL_ROOT": str(install.root) if install.root else "",
        "LUMERICAL_PYTHONPATH": str(install.lumapi_dir) if install.lumapi_dir else "",
        "LUMERICAL_BIN_DIR": str(install.bin_dir) if install.bin_dir else "",
        "ANSYSLMD_LICENSE_FILE": lic or "",
        "status": "ok" if install.root and lic else "partial",
        "messages": "\n".join(messages),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    apply_config_env(config)

    if verbose:
        print("EIDL-Lumapi first-run configuration")
        print(f"  config                    : {path}")
        print(f"  status                    : {config['status']}")
        for message in messages:
            print(f"  - {message}")
        if config["status"] != "ok":
            print("  Some settings are incomplete. Commands will continue with detected values only.")
    return config
