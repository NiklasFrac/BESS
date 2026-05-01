from __future__ import annotations

from pathlib import Path

import yaml


def test_readme_mentions_pipeline_files_and_configured_outputs():
    repo_root = Path(__file__).resolve().parent.parent
    readme = (repo_root / "pv_sim" / "README.md").read_text(encoding="utf-8")
    config = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text(encoding="utf-8"))

    for module_name in [
        "true_pos.py",
        "seen_pos.py",
        "compute_dni.py",
        "compute_poa.py",
        "compute_effective_irradiance.py",
        "modul_sim.py",
        "visualization/horizon_visual.py",
        "visualization/energy_prod_visual.py",
    ]:
        assert module_name in readme

    for path_key in [
        "true_sun_position",
        "apparent",
        "dni",
        "poa",
        "effective_irradiance",
        "energy",
    ]:
        assert config["paths"][path_key] in readme
