#!/usr/bin/env python3
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import DEMO_LOGS, LogAnomalyDetector


def read_cgroup_peak() -> tuple[int | None, str | None]:
    candidates = (
        (Path("/sys/fs/cgroup/memory.peak"), "cgroup_v2"),
        (Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"), "cgroup_v1"),
    )
    for path, source in candidates:
        try:
            raw = path.read_text(encoding="ascii").strip()
            if raw and raw != "max":
                return int(raw), source
        except (OSError, ValueError):
            continue
    return None, None


def main() -> int:
    detector = LogAnomalyDetector().fit()
    batch = [DEMO_LOGS[i % len(DEMO_LOGS)] for i in range(64)]
    for _ in range(8):
        detector.detect_batch(batch)
    time.sleep(5)
    cgroup_peak_bytes, cgroup_source = read_cgroup_peak()
    process_peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "probe": "detector_fit_and_512_inference_records",
        "cgroup_peak_bytes": cgroup_peak_bytes,
        "cgroup_peak_mib": round(cgroup_peak_bytes / (1024 * 1024), 2)
        if cgroup_peak_bytes is not None
        else None,
        "cgroup_source": cgroup_source,
        "process_peak_rss_kib": process_peak_kib,
        "process_peak_rss_mib": round(process_peak_kib / 1024, 2),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
