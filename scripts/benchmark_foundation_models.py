"""
Experimental benchmark harness scaffold for Chronos / TimesFM / Lag-Llama.
This script intentionally keeps dependencies optional and non-blocking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    model: str
    symbol: str
    mape: float
    directional_accuracy: float
    latency_ms: float
    notes: str


def run_placeholder(symbols: list[str]) -> list[BenchmarkResult]:
    # Placeholder numbers for workflow plumbing until model integrations are added.
    out = []
    for sym in symbols:
        out.append(BenchmarkResult("chronos", sym, 5.8, 56.0, 320.0, "placeholder"))
        out.append(BenchmarkResult("timesfm", sym, 5.4, 57.5, 410.0, "placeholder"))
        out.append(BenchmarkResult("lag-llama", sym, 6.2, 55.1, 500.0, "placeholder"))
    return out


if __name__ == "__main__":
    symbols = ["PTT.BK", "AOT.BK", "CPALL.BK"]
    results = run_placeholder(symbols)
    print(json.dumps([asdict(r) for r in results], indent=2))
