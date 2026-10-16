#!/usr/bin/env python
"""运行评测。 python scripts/run_eval.py [--json out.json]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recruiter_agent.eval.benchmark import run_eval  # noqa: E402


if __name__ == "__main__":
    out = None
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "eval_report.json"
    report = run_eval()
    if out:
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已保存: {out}")
