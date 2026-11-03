#!/usr/bin/env python3
"""Measure single-process detector throughput on a repeatable synthetic stream.

Run from the repository root:
    python benchmarks/benchmark.py --records 10000

The result describes this host, Python build, dependency versions, and the
bundled synthetic workload. It is not a production-capacity guarantee.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_LIMIT_VARIABLES:
    os.environ.setdefault(variable, "1")

import numpy
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import DEMO_LOGS, LogAnomalyDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1_000)
    args = parser.parse_args()
    if args.records < 1:
        parser.error("--records must be at least 1")

    detector = LogAnomalyDetector().fit()
    counts = {"NORMAL": 0, "ANOMALY": 0, "IGNORED": 0}
    start = time.perf_counter()
    for index in range(args.records):
        result = detector.detect(DEMO_LOGS[index % len(DEMO_LOGS)])
        counts[result.label] += 1
    elapsed = time.perf_counter() - start
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    report = {
        "records_processed": args.records,
        "elapsed_seconds": round(elapsed, 6),
        "records_per_second": round(args.records / elapsed, 2),
        "classification_counts": counts,
        "process_peak_rss_kib": peak_rss_kib,
        "process_peak_rss_mib": round(peak_rss_kib / 1024, 2),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "workload": "repeated six-record built-in synthetic demonstration stream",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
