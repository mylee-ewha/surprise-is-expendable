import json
from pathlib import Path

def _load_completed(out_path: Path) -> set:
    """이미 완료된 (method, budget) 조합을 results.jsonl에서 읽어 반환."""
    completed = set()
    if not out_path.exists():
        return completed
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add((row["method"], row["kv_budget"]))
            except json.JSONDecodeError:
                pass
    return completed