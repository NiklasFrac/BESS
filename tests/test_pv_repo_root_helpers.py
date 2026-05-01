from __future__ import annotations

from pathlib import Path

import pytest

from pv_sim import compute_dni, compute_effective_irradiance, compute_poa, modul_sim, seen_pos
from pv_sim.visualization import energy_prod_visual, horizon_visual


@pytest.mark.parametrize(
    "module",
    [
        compute_dni,
        compute_effective_irradiance,
        compute_poa,
        modul_sim,
        seen_pos,
        energy_prod_visual,
        horizon_visual,
    ],
)
def test_find_repo_root_finds_parent_data_dir(module, tmp_path: Path):
    (tmp_path / "data").mkdir()
    nested = tmp_path / "pv_sim" / "subdir"
    nested.mkdir(parents=True)

    assert module._find_repo_root(nested) == tmp_path


@pytest.mark.parametrize(
    "module",
    [
        compute_dni,
        compute_effective_irradiance,
        compute_poa,
        modul_sim,
        seen_pos,
        energy_prod_visual,
        horizon_visual,
    ],
)
def test_find_repo_root_raises_without_data_dir(module, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        module._find_repo_root(tmp_path)
