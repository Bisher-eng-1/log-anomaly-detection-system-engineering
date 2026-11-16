from __future__ import annotations

import random

from training_data import ANOMALY_TEMPLATES, NORMAL_TEMPLATES, _render


def build_corpus(
    seed: int = 20261103,
    train_size: int = 320,
    validation_size: int = 120,
    test_normal_size: int = 160,
    test_anomaly_size: int = 160,
):
    rng = random.Random(seed)
    normal_templates = list(NORMAL_TEMPLATES)
    anomaly_templates = list(ANOMALY_TEMPLATES)
    rng.shuffle(normal_templates)
    rng.shuffle(anomaly_templates)

    normal_cut = max(1, int(len(normal_templates) * 0.75))
    anomaly_cut = max(1, int(len(anomaly_templates) * 0.75))

    train_normal_templates = normal_templates[:normal_cut]
    heldout_normal_templates = normal_templates[normal_cut:]
    train_anomaly_templates = anomaly_templates[:anomaly_cut]
    heldout_anomaly_templates = anomaly_templates[anomaly_cut:]

    def make(templates, count, offset):
        return [_render(rng.choice(templates), rng, offset + i) for i in range(count)]

    train_normal = make(train_normal_templates, train_size, 0)
    train_anomaly = make(train_anomaly_templates, train_size, 1_000_000)
    validation = make(heldout_normal_templates, validation_size, 2_000_000)
    test_normal = make(heldout_normal_templates, test_normal_size, 3_000_000)
    test_anomaly = make(heldout_anomaly_templates, test_anomaly_size, 4_000_000)

    test = [(line, 0) for line in test_normal] + [(line, 1) for line in test_anomaly]

    hard_normal = [
        "INFO: exception counter sampled; no active exception",
        "INFO: CPU utilisation is 35 percent",
        "INFO: cache size is 512 MB",
        "INFO: user id 503 logged in",
        "INFO: request completed with HTTP 404 for unknown route",
        "WARN: optional background task delayed by 120 ms",
        "INFO: restart counter is zero",
        "INFO: authentication failed counter remains zero",
        "INFO: processed 512 records successfully",
        "INFO: network connection established with upstream service",
    ]
    hard_anomaly = [
        "INFO: request latency p99 reached 28000 ms and exceeded service objective",
        "INFO: queue backlog increased to 12000 pending messages",
        "INFO: cache hit ratio dropped to 3 percent",
        "INFO: worker throughput dropped below 1 request per second",
        "INFO: thread pool exhausted with zero available workers",
        "INFO: retry rate spiked to 95 percent across the service",
        "INFO: database connection pool saturated with no idle connections",
        "INFO: response time degradation affected 90 percent of requests",
        "INFO: CPU throttling time exceeded 85 percent",
        "INFO: memory pressure caused sustained swap activity",
    ]

    test.extend((line, 0) for line in hard_normal)
    test.extend((line, 1) for line in hard_anomaly)
    rng.shuffle(test)

    return train_normal, train_anomaly, validation, test, {
        "training_normal_templates": len(train_normal_templates),
        "heldout_normal_templates": len(heldout_normal_templates),
        "training_anomaly_templates": len(train_anomaly_templates),
        "heldout_anomaly_templates": len(heldout_anomaly_templates),
    }
