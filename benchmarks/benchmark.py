#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import DEMO_LOGS, LogAnomalyDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.records < 1 or args.batch_size < 1:
        parser.error("--records and --batch-size must be >= 1")

    detector = LogAnomalyDetector().fit()
    stream = [DEMO_LOGS[i % len(DEMO_LOGS)] for i in range(args.records)]

    t0 = time.perf_counter()
    single_counts = {"NORMAL": 0, "ANOMALY": 0, "IGNORED": 0}
    for line in stream:
        single_counts[detector.detect(line).label] += 1
    single_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    batch_counts = {"NORMAL": 0, "ANOMALY": 0, "IGNORED": 0}
    for start in range(0, len(stream), args.batch_size):
        for res in detector.detect_batch(stream[start : start + args.batch_size]):
            batch_counts[res.label] += 1
    batch_elapsed = time.perf_counter() - t1

    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "records_processed": args.records,
        "batch_size": args.batch_size,
        "single_records_per_second": round(args.records / single_elapsed, 2),
        "batch_records_per_second": round(args.records / batch_elapsed, 2),
        "classification_counts": batch_counts,
        "process_peak_rss_mib": round(peak_rss_kib / 1024, 2),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
