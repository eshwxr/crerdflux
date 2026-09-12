"""Phase 4: append-only audit trail for every promotion decision."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.common import ROOT

AUDIT_LOG_PATH = ROOT / "promote" / "audit_log.jsonl"


def log_decision(entry: dict) -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with AUDIT_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log() -> list[dict]:
    if not AUDIT_LOG_PATH.exists():
        return []
    with AUDIT_LOG_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]
