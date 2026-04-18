# src/utils.py
import yaml
import shutil
from pathlib import Path

import yaml

def load_config(config_path):
    encodings = ["utf-8", "utf-8-sig", "gbk"]

    last_error = None
    for enc in encodings:
        try:
            with open(config_path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError as e:
            last_error = e
            continue

    safe_name = Path(config_path).name
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Failed to read config file {safe_name} with encodings: {encodings}"
    )


def prepare_run_dir(work_dir, run_name):
    """Create a clean run directory."""
    path = Path(work_dir) / run_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
