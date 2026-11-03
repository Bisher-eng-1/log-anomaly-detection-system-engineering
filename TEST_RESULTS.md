# Verification report

This report records commands executed in the current verification environment. Host-specific measurements are not portable performance guarantees.

## 1. Automated tests and static checks

Command:

```bash
python -m unittest -v
```

Result: **35 tests passed**.

Coverage includes severity parsing, contextual HTTP status extraction, metadata extraction, minimum training-class validation, classifier probability semantics, classifier-only anomaly detection, guardrail precedence, high-confidence signatures with inconsistent severity prefixes, numeric false-positive protection, bounded input, incremental stream handling, CLI input and custom training files, deterministic evaluation splits, confusion-matrix calculations, and Docker memory-size parsing.

Python bytecode compilation also completed successfully:

```bash
python -m compileall -q app.py training_data.py evaluation tests scripts benchmarks
```

## 2. Functional execution

The following commands completed successfully:

```bash
python app.py --demo
python app.py --input samples/sample.log
```

The six bundled demonstration records were classified as follows:

| Record | Final label | Decision source |
|---|---|---|
| `INFO: System running` | NORMAL | random_forest_classifier |
| `INFO: health check passed` | NORMAL | random_forest_classifier |
| `WARN: High CPU` | ANOMALY | operational_guardrail |
| `INFO: request completed with HTTP 200` | NORMAL | random_forest_classifier |
| `ERROR: HTTP 500 Internal Server Error` | ANOMALY | operational_guardrail |
| `INFO: worker is ready` | NORMAL | random_forest_classifier |

## 3. Synthetic held-out model evaluation

Command:

```bash
python -m evaluation.evaluate_model --output evaluation/results.json
```

The evaluation used 320 labelled normal training records and 320 labelled anomaly training records. Training and test data used disjoint message-template groups. The held-out test set contained 170 normal and 170 anomaly records, including 10 challenge examples in each class.

| Metric | Random Forest only | Classifier + guardrails |
|---|---:|---:|
| True positives | 125 | 161 |
| True negatives | 168 | 168 |
| False positives | 2 | 2 |
| False negatives | 45 | 9 |
| Precision | 98.43% | 98.77% |
| Recall | 73.53% | 94.71% |
| F1-score | 84.18% | 96.70% |
| False-positive rate | 1.18% | 1.18% |
| Accuracy | 86.18% | 96.76% |

On the separate normal-only validation set, the classifier produced 2 false positives from 120 records (1.67%) at the fixed 0.50 probability threshold.

**Interpretation:** the classifier alone prioritised precision and missed some synthetic failure patterns. The explicit guardrail layer recovered 36 additional positive cases in this particular test corpus. The remaining 9 false negatives show that the hybrid rules still do not cover every operational anomaly. The result supports the implementation's testability; it does not establish real-world detection accuracy. The dataset is synthetic, limited, and deliberately bounded. Real service logs may produce materially different results.

The complete per-run report, including seed, split sizes, software versions, and limitations, is stored in `evaluation/results.json`.

## 4. Throughput and host process RSS

Command:

```bash
python benchmarks/benchmark.py --records 1000
```

Observed in this environment:

| Measurement | Result |
|---|---:|
| Records processed | 1,000 |
| Elapsed inference time | 9.618213 seconds |
| Throughput | 103.97 records/second |
| Process peak RSS | 166,984 KiB (163.07 MiB) |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |

The benchmark repeatedly processes the six-record built-in synthetic demonstration stream. The result is a host process measurement, not Docker cgroup memory, and is not a production throughput guarantee.

## 5. Docker verification status

The Dockerfile, Compose resource settings, and `scripts/validate_docker.py` are included. The configured memory limit is **240 MiB**, with memory-plus-swap also set to 240 MiB.

Docker validation was **not executed in this environment** because the Docker CLI/daemon is unavailable. Consequently, this report makes no claim that the image has been built, that the workload has run inside Docker, that the container stayed below the configured ceiling, or that an actual cgroup peak was measured.

Run the following on a Docker-enabled host:

```bash
python scripts/validate_docker.py
```

A successful report must show `result: PASS`, `oom_killed: false`, the expected 240 MiB configured limit, and a non-null observed container memory peak. Keep the generated `docker-validation-results.json` only after inspecting the actual output.

## 6. Verification boundary

- Unit tests establish the tested code paths, not production correctness.
- Synthetic evaluation establishes metrics only for the generated corpus.
- Host RSS is not a container memory measurement.
- A configured Docker memory ceiling is not evidence that the workload fits beneath it.
- Production claims require representative labelled telemetry, a documented evaluation split, and measurements from the target container runtime.
