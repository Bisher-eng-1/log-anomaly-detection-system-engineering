"""Deterministic synthetic training examples for the standalone prototype.

Replace these fixtures with reviewed, labelled service logs before deployment.
No real customer or production log data is included in this repository.
"""
from __future__ import annotations

import random

BASELINE_NORMALS = (
    "INFO: service started successfully", "INFO: health check passed",
    "INFO: request completed with HTTP 200", "INFO: database connection pool ready",
    "INFO: cache lookup completed", "INFO: scheduled task completed",
    "INFO: worker is ready", "INFO: system running normally",
    "INFO: configuration loaded", "INFO: heartbeat received",
    "INFO: message processed successfully", "INFO: connection closed normally",
    "INFO: request handled in 24 ms", "INFO: queue depth is 3 messages",
    "INFO: cache size is 512 MB", "INFO: processed 512 records",
    "INFO: user 503 logged in", "INFO: database query completed in 18 ms",
    "INFO: background job completed", "INFO: scheduled backup completed",
    "INFO: file uploaded successfully", "INFO: session created",
    "INFO: session expired normally", "INFO: API request completed with status=200",
    "INFO: connection established to database", "INFO: message acknowledged by consumer",
    "INFO: batch of 128 records committed", "INFO: application configuration refreshed",
    "INFO: cache hit for product catalog", "INFO: cache miss; value loaded from database",
    "DEBUG: retry counter initialised to zero", "DEBUG: request headers validated",
    "DEBUG: database statement prepared", "DEBUG: worker entered idle state",
    "DEBUG: connection returned to pool", "DEBUG: request correlation id assigned",
    "WARN: retrying a transient request after backoff", "WARN: optional cache refresh was delayed",
    "WARN: non-critical background task will retry", "WARN: client supplied an unsupported optional field",
    "WARN: request latency briefly exceeded the soft target",
    "WARN: one message was retried and then acknowledged",
    "WARN: optional report export was delayed", "WARN: cache entry expired and will be refreshed",
    "WARN: optional report export was skipped by configuration",
    "WARN: one background notification was deferred",
    "DEBUG: diagnostic counters were flushed", "DEBUG: worker completed its periodic status check",
    "INFO: CPU utilisation is 32 percent", "INFO: memory usage is 48 percent of the container limit",
    "INFO: disk utilisation is 61 percent", "INFO: network connection established with the upstream service",
    "INFO: timeout budget configured to 2000 ms", "DEBUG: exception counter sampled; no active exception",
    "INFO: restart counter is zero", "INFO: authentication failed counter remains zero",
    "INFO: request completed with HTTP 404 for an unknown route",
    "WARN: client request returned HTTP 400 with a validation message",
)

NORMAL_TEMPLATES = (
    "INFO: request completed with HTTP 200 in {latency} ms",
    "INFO: database query completed in {latency} ms",
    "INFO: worker {worker} processed {count} messages successfully",
    "INFO: cache {cache_action} for key {key}",
    "INFO: CPU utilisation is {cpu} percent",
    "INFO: memory usage is {memory} percent of the container limit",
    "INFO: disk utilisation is {disk} percent",
    "INFO: queue depth is {queue} messages",
    "DEBUG: connection {connection} returned to pool",
    "INFO: health check passed for service {service}",
    "INFO: batch {batch} committed successfully",
    "WARN: optional background task delayed by {delay} ms",
    "WARN: transient request retried after {delay} ms backoff",
    "INFO: HTTP 404 returned for unknown route {route}",
    "INFO: scheduled task {task} completed",
    "DEBUG: correlation id {key} attached to request",
)
ANOMALY_TEMPLATES = (
    "FATAL: kernel panic detected on worker {worker}",
    "CRITICAL: out of memory; process killed by allocator {key}",
    "ERROR: segmentation fault in native module {module}",
    "ERROR: database corruption detected in table {table}",
    "ERROR: HTTP 503 upstream service unavailable after {latency} ms",
    "ERROR: connection refused by primary database on port {port}",
    "CRITICAL: disk full; filesystem read-only on volume {volume}",
    "ERROR: unrecoverable exception in payment worker {worker}",
    "WARN: high CPU utilisation reached {cpu} percent",
    "WARN: high memory pressure reached {memory} percent",
    "ERROR: request timed out after {latency} ms; upstream unreachable",
    "FATAL: repeated worker crash; restart limit exceeded for {worker}",
    "INFO: database corruption detected in table {table}",
    "INFO: memory allocation failure terminated worker {worker}",
    "INFO: segmentation fault in native module {module}",
    "INFO: primary database connection refused on port {port}",
    "INFO: storage volume read-only after disk full condition {volume}",
    "INFO: unrecoverable exception stopped service {service}",
)


def _render(template: str, rng: random.Random, index: int) -> str:
    values = {
        "latency": rng.randint(5, 30_000), "worker": f"w{rng.randint(1, 80)}",
        "count": rng.randint(1, 500), "cache_action": rng.choice(("hit", "miss", "refresh")),
        "key": f"k{index:06d}", "cpu": rng.randint(5, 99),
        "memory": rng.randint(15, 99), "disk": rng.randint(10, 99),
        "queue": rng.randint(0, 45), "connection": f"c{rng.randint(1, 100)}",
        "service": rng.choice(("api", "worker", "catalog", "billing")),
        "batch": rng.randint(1, 10_000), "delay": rng.randint(5, 1_500),
        "route": f"/unknown/{rng.randint(1, 900)}",
        "task": rng.choice(("cleanup", "refresh", "report", "backup")),
        "module": rng.choice(("parser", "codec", "storage", "native_ext")),
        "table": rng.choice(("orders", "sessions", "events", "accounts")),
        "port": rng.choice((5432, 3306, 6379)), "volume": rng.choice(("data", "logs", "cache")),
    }
    return template.format(**values)


def build_training_data(seed: int = 20261103, normal_count: int = 320,
                         anomaly_count: int = 320) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    records: list[str] = []
    labels: list[int] = []
    for line in BASELINE_NORMALS:
        records.append(line)
        labels.append(0)
    for index in range(normal_count):
        records.append(_render(rng.choice(NORMAL_TEMPLATES), rng, index))
        labels.append(0)
    for index in range(anomaly_count):
        records.append(_render(rng.choice(ANOMALY_TEMPLATES), rng, 1_000_000 + index))
        labels.append(1)
    order = list(range(len(records)))
    rng.shuffle(order)
    return [records[i] for i in order], [labels[i] for i in order]


def build_default_anomaly_examples(count: int = 320, seed: int = 20261103) -> list[str]:
    rng = random.Random(seed + 17)
    return [
        _render(rng.choice(ANOMALY_TEMPLATES), rng, 2_000_000 + index)
        for index in range(count)
    ]


DEFAULT_ANOMALY_TRAINING = build_default_anomaly_examples()
