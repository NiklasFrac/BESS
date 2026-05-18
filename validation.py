import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())
    raw = repo_root / cfg["paths"]["pvgis_raw"]
    out = repo_root / cfg["paths"]["pvgis"]
    report = repo_root / cfg["paths"]["validation_report"]

    df = pd.read_csv(raw)
    replaced = int(df["horizon_height_deg"].isna().sum())
    df["horizon_height_deg"] = df["horizon_height_deg"].fillna(0)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"pvgis": {"file": str(raw), "horizon_height_deg_nan_replaced": replaced}}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
