from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TASK_FILE_MAP = {
    "easy": "easy_task.json",
    "medium": "medium_task.json",
    "hard": "hard_task.json",
}


def load_task(task_name: str):
    normalized = task_name.strip().lower()
    if normalized not in TASK_FILE_MAP:
        raise ValueError(f"Unknown task name: {task_name}")

    task_file = DATA_DIR / TASK_FILE_MAP[normalized]
    if not task_file.exists():
        raise FileNotFoundError(f"Task file does not exist: {task_file}")

    with task_file.open("r", encoding="utf-8") as f:
        return json.load(f)
