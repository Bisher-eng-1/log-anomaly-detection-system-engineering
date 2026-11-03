#!/usr/bin/env python3
"""Build, run, and measure the log detector under a Docker memory limit.

Run from the repository root with:
    python scripts/validate_docker.py

The script writes docker-validation-results.json with measurements from the
current host. It never substitutes estimated values for observed measurements.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "log-anomaly-detection-system:validation"
CONTAINER = "log-anomaly-memory-validation"
MEMORY_LIMIT_BYTES = 240 * 1024 * 1024
REPORT_PATH = ROOT / "docker-validation-results.json"


def run(command: list[str], *, capture: bool = False, check: bool = True):
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=check,
    )


def parse_size(value: str) -> int | None:
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)\s*$", value, re.I)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    powers = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    return int(number * powers[unit])


def stats_memory_bytes() -> int | None:
    result = run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
        capture=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    used = result.stdout.strip().split("/", 1)[0].strip()
    return parse_size(used)


def main() -> int:
    if shutil.which("docker") is None:
        print("ERROR: Docker CLI was not found. Install Docker Engine/Desktop and retry.", file=sys.stderr)
        return 2

    version = run(["docker", "version", "--format", "{{.Server.Version}}"], capture=True, check=False)
    if version.returncode != 0:
        print("ERROR: Docker daemon is unavailable or the current user lacks permission.", file=sys.stderr)
        if version.stdout:
            print(version.stdout.strip(), file=sys.stderr)
        return 2

    print(f"[1/4] Building {IMAGE}")
    run(["docker", "build", "-t", IMAGE, "."])

    print("[2/4] Running the built-in demo with a 240 MiB memory and swap ceiling")
    demo = run(
        ["docker", "run", "--rm", "--memory=240m", "--memory-swap=240m", IMAGE],
        capture=True,
        check=False,
    )
    print(demo.stdout.rstrip())
    if demo.returncode != 0:
        print(f"ERROR: constrained demo exited with status {demo.returncode}.", file=sys.stderr)
        return demo.returncode or 1

    print("[3/4] Measuring model initialisation and inference inside a constrained container")
    run(["docker", "rm", "--force", CONTAINER], check=False, capture=True)
    run(
        [
            "docker", "run", "--detach", "--name", CONTAINER,
            "--memory=240m", "--memory-swap=240m", IMAGE,
            "python", "scripts/container_memory_probe.py",
        ]
    )

    samples: list[int] = []
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = run(
            ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER],
            capture=True,
            check=False,
        )
        if state.returncode != 0 or state.stdout.strip().lower() != "true":
            break
        sample = stats_memory_bytes()
        if sample is not None:
            samples.append(sample)
        time.sleep(0.25)

    run(["docker", "wait", CONTAINER], capture=True, check=False)
    logs = run(["docker", "logs", CONTAINER], capture=True, check=False)
    inspect = run(
        [
            "docker", "inspect", "--format",
            "{{.State.OOMKilled}} {{.State.ExitCode}} {{.HostConfig.Memory}}",
            CONTAINER,
        ],
        capture=True,
        check=False,
    )
    run(["docker", "rm", "--force", CONTAINER], check=False, capture=True)

    print(logs.stdout.rstrip())
    fields = inspect.stdout.strip().split() if inspect.returncode == 0 else []
    if len(fields) != 3:
        print("ERROR: Could not read container exit and OOM state.", file=sys.stderr)
        return 1
    oom_killed = fields[0].lower() == "true"
    exit_code = int(fields[1])
    configured_limit = int(fields[2])

    probe_report = None
    for line in reversed(logs.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("probe"):
            probe_report = candidate
            break

    sampled_peak = max(samples) if samples else None
    cgroup_peak = probe_report.get("cgroup_peak_bytes") if probe_report else None
    measured_candidates = [value for value in (sampled_peak, cgroup_peak) if isinstance(value, int)]
    observed_peak = max(measured_candidates) if measured_candidates else None
    result = {
        "image": IMAGE,
        "docker_server_version": version.stdout.strip(),
        "memory_limit_bytes": configured_limit,
        "memory_limit_mib": round(configured_limit / (1024 * 1024), 2),
        "demo_exit_code": demo.returncode,
        "probe_exit_code": exit_code,
        "oom_killed": oom_killed,
        "docker_stats_sample_count": len(samples),
        "docker_stats_peak_bytes": sampled_peak,
        "docker_stats_peak_mib": round(sampled_peak / (1024 * 1024), 2) if sampled_peak is not None else None,
        "cgroup_peak_bytes": cgroup_peak,
        "cgroup_peak_mib": probe_report.get("cgroup_peak_mib") if probe_report else None,
        "process_peak_rss_mib": probe_report.get("process_peak_rss_mib") if probe_report else None,
        "observed_peak_bytes": observed_peak,
        "observed_peak_mib": round(observed_peak / (1024 * 1024), 2) if observed_peak is not None else None,
        "result": (
            "PASS"
            if demo.returncode == 0
            and exit_code == 0
            and not oom_killed
            and configured_limit == MEMORY_LIMIT_BYTES
            and observed_peak is not None
            and observed_peak <= configured_limit
            else "FAIL"
        ),
        "measurement_note": (
            "Docker stats is sampled; cgroup peak is preferred when available. "
            "Values are host- and workload-specific."
        ),
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("[4/4] Validation summary")
    print(json.dumps(result, indent=2))
    print(f"Report written to: {REPORT_PATH.relative_to(ROOT)}")

    if observed_peak is None:
        print(
            "ERROR: No usable container memory peak was observed; validation is incomplete.",
            file=sys.stderr,
        )
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
