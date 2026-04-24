import logging
from pathlib import Path

import yaml

from pv_sim.compute_dni import main as compute_dni_main
from pv_sim.compute_effective_irradiance import main as compute_effective_irradiance_main
from pv_sim.compute_poa import main as compute_poa_main
from pv_sim.modul_sim import main as pv_sim_main
from pv_sim.seen_pos import main as seen_pos_main
from pv_sim.true_pos import main as true_pos_main
from pv_sim.visualization.energy_prod_visual import main as energy_visual_main
from pv_sim.visualization.horizon_visual import main as horizon_visual_main


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Repo-Root nicht gefunden. Erwartet ein Verzeichnis mit 'data'-Ordner."
    )


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    log_cfg = cfg["logging"]
    logging.basicConfig(
        level=log_cfg["level"],
        format=log_cfg["format"],
        datefmt=log_cfg["datefmt"],
    )
    log = logging.getLogger(__name__)

    steps = [
        ("True Solar Position", true_pos_main),
        ("Seen Solar Position", seen_pos_main),
        ("Compute DNI", compute_dni_main),
        ("Compute POA", compute_poa_main),
        ("Compute Effective Irradiance", compute_effective_irradiance_main),
        ("PV Simulation", pv_sim_main),
        ("Horizon Visualisation", horizon_visual_main),
        ("Energy Visualisation", energy_visual_main),
    ]

    for name, fn in steps:
        log.info("=== %s ===", name)
        fn()


if __name__ == "__main__":
    main()
