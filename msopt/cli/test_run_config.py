from pathlib import Path

from msopt.cli.run import _isolate_lumerical_config


def test_isolate_lumerical_config_copies_resource_settings(tmp_path: Path):
    home = tmp_path / "home"
    source = home / ".config" / "Lumerical" / "FDTD Solutions.ini"
    source.parent.mkdir(parents=True)
    source.write_text("[jobmanager]\nFDTD_v2=<engines />\n")
    env = {"HOME": str(home)}
    run_dir = tmp_path / "run"

    target_root = _isolate_lumerical_config(env, run_dir)

    copied = target_root / "Lumerical" / "FDTD Solutions.ini"
    assert copied.read_text() == source.read_text()
    assert env["HOME"] == str(home)
    assert env["XDG_CONFIG_HOME"] == str(target_root)


def test_isolate_lumerical_config_honors_existing_xdg_root(tmp_path: Path):
    source_root = tmp_path / "source-xdg"
    source = source_root / "Lumerical" / "FDTD Solutions.ini"
    source.parent.mkdir(parents=True)
    source.write_text("resource config")
    env = {"HOME": str(tmp_path / "home"), "XDG_CONFIG_HOME": str(source_root)}

    target_root = _isolate_lumerical_config(env, tmp_path / "run")

    assert (target_root / "Lumerical" / source.name).read_text() == "resource config"
