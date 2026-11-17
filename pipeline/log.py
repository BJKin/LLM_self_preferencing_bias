import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def log_response(path: Path, response: dict, **fields) -> None:
    """ Append one model response to a JSONL file """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **fields, **response}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_log(path: Path) -> pd.DataFrame:
    """ Read a JSONL log back into a DataFrame """
    df = pd.read_json(path, lines=True, dtype=False)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
