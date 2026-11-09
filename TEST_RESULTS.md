Verification Report
This report records commands executed in the verification environment. Host-specific measurements are not portable performance guarantees.

1. Automated Tests and Static Checks
Command:

python -m unittest -v
Result: 35 tests passed.

Coverage includes severity parsing, contextual HTTP status extraction, metadata extraction, minimum training-class validation, classifier probability semantics, classifier-only anomaly detection, guardrail precedence, high-confidence signatures with inconsistent severity prefixes, numeric false-positive protection, bounded input, incremental stream handling, CLI input and custom training files, deterministic evaluation splits, confusion-matrix calculations, and Docker memory-size parsing.

Python bytecode compilation also completed successfully:

python -m compileall -q app.py training_data.py evaluation tests scripts benchmarks
2. Functional Execution
The following commands completed successfully:

python app.py --demo
python app.py --input samples/sample.log
The six bundled demonstration records were classified as follows:

Record	Final label	Decision source
INFO: System running	NORMAL	random_forest_classifier
INFO: health check passed	NORMAL	random_forest_classifier
WARN: High CPU	ANOMALY	operational_guardrail
INFO: request completed with HTTP 200	NORMAL	random_forest_classifier
ERROR: HTTP 500 Internal Server Error	ANOMALY	operational_guardrail
INFO: worker is ready	NORMAL	random_forest_classifier
3. Synthetic Held-Out Model Evaluation
Command:

python -m evaluation.evaluate_model --output evaluation/results.json
The evaluation used 320 labelled normal training records and 320 labelled anomaly training records. Training and test data used disjoint message-template groups. The held-out test set contained 170 normal and 170 anomaly records, including 10 challenge examples in each class.

Metric	Random Forest only	Classifier + guardrails
True positives	125	161
True negatives	168	168
False positives	2	2
False negatives	45	9
Precision	98.43%	98.77%
Recall	73.53%	94.71%
F1-score	84.18%	96.70%
False-positive rate	1.18%	1.18%
Accuracy	86.18%	96.76%
On the separate normal-only validation set, the classifier produced 2 false positives from 120 records (1.67%) at the fixed 0.50 probability threshold.

Interpretation: The classifier alone prioritised precision and missed some synthetic failure patterns. The explicit guardrail layer recovered 36 additional positive cases in this particular test corpus. The remaining 9 false negatives show that the hybrid rules still do not cover every operational anomaly. The result supports the implementation's testability; it does not establish real-world detection accuracy. The dataset is synthetic, limited, and deliberately bounded. Real service logs may produce materially different results.

The complete per-run report, including seed, split sizes, software versions, and limitations, is stored in evaluation/results.json.

4. Throughput and Process RSS
4.1 Host benchmark
Command:

python benchmarks/benchmark.py --records 1000
Previously recorded host measurement:

Measurement	Result
Records processed	1,000
Elapsed inference time	9.618213 seconds
Throughput	103.97 records/second
Process peak RSS	166,984 KiB (163.07 MiB)
Python	3.13.5
NumPy	2.3.5
SciPy	1.17.0
scikit-learn	1.8.0
4.2 Docker benchmark
Command:

docker compose run --rm --no-deps --entrypoint python log-anomaly-detector benchmarks/benchmark.py --records 10000
Observed inside the Docker container:

Measurement	Result
Records processed	10,000
Elapsed time	460.327783 seconds
Throughput	21.72 records/second
Process peak RSS	125,272 KiB (122.34 MiB)
Python	3.11.18
NumPy	2.3.5
SciPy	1.17.0
scikit-learn	1.8.0
The benchmark repeatedly processes the six-record built-in synthetic demonstration stream. These measurements describe this specific container environment and workload. They are not production throughput guarantees.

5. Docker Build, Execution, and Memory Measurements
5.1 Image build
Command:

docker compose build
Result: Build completed successfully.

The image log-anomaly-detection-system:local was built successfully. The Docker build included the application, sample data, scripts, and benchmark directory.

5.2 Demo execution
The Docker Compose demo completed successfully with exit code 0.

The output contained six classified demonstration records, including normal records and operational anomalies.

5.3 Configured memory limit
Docker Compose configuration:

Memory limit: 240 MiB

Memory plus swap limit: 240 MiB

CPU limit: 1.0

Read-only container filesystem

Temporary filesystem: 16 MiB

The container configuration was inspected using Docker:

docker inspect log-anomaly-detection-system-log-anomaly-detector-1 --format "Memory Limit: {{.HostConfig.Memory}} bytes"
Observed configured memory limit:

251658240 bytes
This equals 240 MiB.

5.4 Container memory probe
Command:

docker compose run --rm --no-deps --entrypoint python log-anomaly-detector -m scripts.container_memory_probe
Observed output:

Measurement	Result
Probe workload	Model fit + 500 inference records
Reported cgroup peak	82.05 MiB
Process peak RSS	123.65 MiB
cgroup source	cgroup v2
Measurement limitation: The reported cgroup peak is lower than the process peak RSS, although both were reported by the same probe execution. These measurements are not directly consistent and require further investigation before the cgroup value can be treated as a reliable container-wide memory peak. The configured 240 MiB limit is verified, but this report does not use the 82.05 MiB figure as conclusive proof of total container memory consumption.

6. Verification Boundaries
Unit tests establish the tested code paths, not production correctness.

Synthetic evaluation establishes metrics only for the generated corpus.

Benchmark throughput is specific to the tested machine, container, software versions, and workload.

Process RSS and cgroup memory measurements are different metrics and must not be treated as interchangeable.

The Docker memory ceiling was verified as configured, but the inconsistent memory probe readings require further investigation before making a definitive container memory-usage claim.

Production claims require representative labelled telemetry, a documented evaluation split, and measurements from the target container runtime.