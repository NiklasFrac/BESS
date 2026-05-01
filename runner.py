from pathlib import Path
import logging

import pandas as pd
import yaml

from download.run_downloads import main as run_downloads_main
from pv_sim.run_pv import main as run_pv_main
from battery_sim.simulator import simulate as simulate_battery


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "config.yaml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Repo-Root nicht gefunden.")


def load_config(repo_root: Path) -> dict:
    return yaml.safe_load(
        (repo_root / "configs" / "config.yaml").read_text(encoding="utf-8")
    )

def setup_logging(config: dict) -> None:
    log_cfg = config["logging"]
    logging.basicConfig(
        level=log_cfg["level"],
        format=log_cfg["format"],
        datefmt=log_cfg["datefmt"],
    )


def run_battery(repo_root: Path, config: dict) -> pd.DataFrame:
    log = logging.getLogger(__name__)

    log.info("=== Battery Simulation ===")

    results = pd.DataFrame(simulate_battery(repo_root, config))

    if results.empty:
        raise RuntimeError("Battery simulation returned empty results.")

    output_path = repo_root / config["paths"]["battery_results"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    log.info("Battery results written to: %s", output_path)
    return results


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    config = load_config(repo_root)
    setup_logging(config)

    log = logging.getLogger(__name__)

    log.info("=== Full Pipeline Start ===")

    run_downloads_main()
    run_pv_main()
    run_battery(repo_root, config)

    log.info("=== Full Pipeline Done ===")


if __name__ == "__main__":
    main()