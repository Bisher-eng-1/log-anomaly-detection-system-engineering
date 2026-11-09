# Log Anomaly Detection System

A resource-constrained, streaming log-analysis prototype built with
Python and scikit-learn. It combines a supervised Random Forest
classifier with explicit operational guardrails, sparse text/metadata
features, bounded input processing, Docker memory controls, and
reproducible evaluation tooling.

## Engineering goals

-   Process log records incrementally rather than loading an entire
    input log into memory.
-   Classify each record using a trained machine-learning model and
    report the model's anomaly-probability estimate.
-   Apply deterministic guardrails to selected high-confidence
    operational failures.
-   Keep feature matrices sparse and record size bounded.
-   Run as a non-root Docker container with a configured memory ceiling
    below 256 MiB.
-   Make tests, synthetic evaluation, throughput measurements, and
    container-memory validation repeatable.

## Architecture

``` mermaid
flowchart TD
    A[stdin or log file] --> B[Bounded line reader]
    B --> C[Record normalisation]
    C --> D[Feature extraction]
    D --> E[HashingVectorizer: word uni/bi-grams]
    D --> F[DictVectorizer: operational metadata]
    E --> G[CSR sparse feature matrix]
    F --> G
    G --> H[Random Forest classifier]
    H --> I[Anomaly probability estimate]
    C --> J[Operational guardrails]
    I --> K{Final decision}
    J --> K
    K --> L[NORMAL / ANOMALY / IGNORED]
    L --> M[stdout: score, label, detection source]
```

### Processing stages

1\. **Bounded input:** standard input and file input are consumed one
physical line at a time. A record is limited to 16,384 characters. When
an oversized physical line is encountered, its remainder is discarded
before the next record is read. 2. **Text features:**
`HashingVectorizer` creates a fixed-width representation with 1,024
hashed word features and unigram/bigram tokens. Hashing avoids
maintaining a growing vocabulary. 3. **Operational metadata:**
`DictVectorizer` encodes severity and indicators for HTTP errors,
exceptions, timeouts, CPU, memory, disk/storage, authentication
failures, restarts, and network faults. 4. **Sparse representation:**
text and metadata features are joined into a CSR sparse matrix using
SciPy. 5. **Classifier:** a `RandomForestClassifier` is trained on
labelled normal and anomaly examples. The default model is fitted once
when the process starts. 6. **Operational guardrails:** explicit rules
override the classifier for selected cases such as HTTP 5xx,
resource-pressure warnings, and fatal failure signatures. 7. **Output:**
each non-empty record produces a label, a signed classifier margin, a
class-1 probability estimate, and the source that determined the final
label.

## Machine-learning design

### Feature representation

  -----------------------------------------------------------------------
  Component               Configuration           Purpose
  ----------------------- ----------------------- -----------------------
  `HashingVectorizer`     1,024 features;         Fixed-size text
                          `(1, 2)` n-grams;       representation
                          lowercase; non-negative 
                          hashing                 

  `DictVectorizer`        Sparse output           Severity and
                                                  operational indicators

  Feature assembly        SciPy CSR matrix        Avoids dense
                                                  text-feature allocation
  -----------------------------------------------------------------------

### Classifier configuration

  Parameter                                                Value
  -------------------------- -----------------------------------
  Estimators                                           240 trees
  Maximum depth                                               14
  Minimum samples per leaf                                     2
  Feature sampling                                        `sqrt`
  Class weighting                           `balanced_subsample`
  Parallel jobs                                                1
  Random state                                     42 by default
  Classification threshold     0.50 class-1 probability estimate

`model_anomaly_score` is `predict_proba(...)[class=1]`. It is a model
probability estimate, not a calibrated production risk probability.
`decision_value` is `0.5 - model_anomaly_score`; negative values
indicate that the classifier crossed the anomaly threshold. Guardrails
can still raise an alert when the classifier probability is below 0.50.

The bundled training corpus is deterministic and synthetic. It includes
ordinary baseline messages, generated normal records, and generated
anomaly records. It makes the example executable without private
production logs; it does not establish how the classifier will perform
on an unseen organisation's log format. Before deployment, train and
evaluate with representative, reviewed, labelled service logs.

### Custom labelled training data

Supply one normal-only file and one anomaly-only file. Each physical
line is one labelled record:

``` bash
python app.py \\
  --normal-training data/normal.log \\
  --anomaly-training data/anomalies.log \\
  --input samples/sample.log
```

Both training arguments are required together. The files are read at
startup; they are not retained by the application. Do not use unreviewed
or sensitive production logs in a public repository.

## Operational guardrails

The deterministic layer currently covers:

-   HTTP status codes from 500 through 599, but only when the code
    appears in an explicit HTTP/status context such as `HTTP 503`,
    `status_code=503`, or `response status: 503`.
-   `WARN` records containing selected resource-pressure phrases,
    including high CPU, high memory, memory exhausted, disk full, and
    resource exhausted.
-   `CRITICAL` and `FATAL` severity records.
-   High-confidence failure phrases such as kernel panic, segmentation
    fault, out-of-memory, OOM, database corruption, disk full,
    connection refused, and unrecoverable failures, even when the log's
    severity prefix is inconsistent.
-   Additional selected `ERROR` signatures such as exception, traceback,
    and authentication failure.

A bare number such as `503` in a user ID or `512` in a record count is
not treated as an HTTP status. The guardrail list is intentionally
explicit and should be adapted to the target service's log conventions.

## Repository structure

``` text
log-anomaly-detection-system/
├── app.py
├── training_data.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── README.md
├── TEST_RESULTS.md
├── docs/
│   ├── TECHNICAL_WALKTHROUGH.md
│   └── VALIDATION_CHECKLIST.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── samples/
│   └── sample.log
├── evaluation/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate_model.py
│   └── results.json
├── tests/
│   ├── __init__.py
│   ├── test_app.py
│   ├── test_docker_validation.py
│   └── test_evaluation.py
├── scripts/
│   ├── __init__.py
│   ├── container_memory_probe.py
│   └── validate_docker.py
└── benchmarks/
    └── benchmark.py
```

## Local setup

-   Python 3.11 or later
-   Dependencies pinned in `requirements.txt`
-   Docker Engine for image build and cgroup memory validation

``` bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

Built-in demonstration:

``` bash
python app.py --demo
```

Process the bundled sample file:

``` bash
python app.py --input samples/sample.log
```

Read a live stream from standard input:

``` bash
python app.py
```

Provide one record per line. End input with `Ctrl+D` on Linux/macOS or
`Ctrl+Z` followed by Enter on Windows.

## Automated tests

Run from the repository root:

``` bash
python -m unittest -v
```

The suite covers severity parsing, contextual HTTP status extraction,
metadata features, training-data validation, classifier probability
semantics, classifier-only anomaly detection, guardrail precedence,
numeric false-positive cases, bounded input, incremental stream
processing, CLI behaviour, evaluation metrics, deterministic dataset
splitting, and Docker memory-size parsing.

## Model evaluation

Run the reproducible offline evaluation:

``` bash
python -m evaluation.evaluate_model --output evaluation/results.json
```

The evaluator trains on one group of message templates and evaluates on
held-out templates, with additional challenge records that contain
benign keyword/numeric mentions and anomalies without obvious severity
prefixes. It reports the classifier-only and hybrid confusion matrices,
precision, recall, F1, false-positive rate, and accuracy.

**Important:** this is a deterministic synthetic benchmark, not a
production dataset. Its metrics describe only the included generated
workload. They must not be represented as expected real-world accuracy.
A credible deployment evaluation requires a separately labelled dataset
collected from the target services, with time-aware or service-aware
splits to avoid leakage.

The checked-in `evaluation/results.json` is generated by the command
above and records the seed, split sizes, environment versions, and
limitations.

## Throughput and process-memory benchmark

``` bash
python benchmarks/benchmark.py --records 1000
```

The benchmark reports the elapsed inference time, records per second,
classification counts, process peak RSS, and dependency versions. It
repeatedly processes the six-record built-in demonstration stream. This
is a microbenchmark of the current host and workload, not a production
capacity claim. Process RSS is not the same measurement as Docker cgroup
memory.

## Docker deployment and resource limit

The image uses Python 3.11 slim, runs as a non-root user, and installs
pinned binary Python dependencies. The container configuration applies a
**240 MiB memory limit** and an equal memory-plus-swap ceiling. The
tested workloads completed successfully under this limit without an
out-of-memory termination.

  Measurement                                                      Result
  ---------------------------------------------- ------------------------
  Configured Docker memory limit                                  240 MiB
  Process peak RSS --- memory probe                            123.65 MiB
  Process peak RSS --- 10,000-record benchmark                 122.34 MiB
  Memory-probe execution                           Completed successfully
  Docker benchmark                                 Completed successfully

The memory probe exercised model training followed by 500 inference
records. A separate benchmark processed 10,000 records inside the
constrained container. These results demonstrate successful execution of
the tested workloads under the configured 240 MiB ceiling; they are not
a guarantee for arbitrary workloads or deployment environments.

**Measurement note:** Process RSS and cgroup memory counters reported
different values during validation. They are distinct measurement
sources and should not be treated as interchangeable. The process RSS
measurements are reported separately rather than being presented as
container-wide cgroup peak usage.

Build and run:

``` bash
docker build -t log-anomaly-detection-system .
docker run --rm \\
  --memory=240m --memory-swap=240m \\
  log-anomaly-detection-system
```

Or use Compose:

``` bash
docker compose build
docker compose up --abort-on-container-exit
```

Run against a mounted sample:

``` bash
docker run --rm \\
  --memory=240m --memory-swap=240m \\
  -v "$PWD/samples\:/data:ro" \\
  log-anomaly-detection-system \\
  python app.py --input /data/sample.log
```

### Actual container-memory validation

Run this on a host with a working Docker daemon:

``` bash
python scripts/validate_docker.py
```

The validation script builds the image, runs the demo under the
configured ceiling, starts a separate probe that trains the model and
performs inference, samples `docker stats`, reads the cgroup v2/v1 peak
counter when available, and checks the configured memory limit and
OOM-kill state. It writes `docker-validation-results.json` with the
measurements from that machine.

The script requires an observed memory value to report `PASS`. If the
runtime does not expose a usable cgroup peak and Docker stats cannot be
sampled, it reports an incomplete/failed validation rather than
inventing a number. Container memory is host-, runtime-, and
workload-specific.

## Known limitations and next engineering steps

-   The bundled training and evaluation data are synthetic; real-world
    model quality is not established.
-   The classifier learns the patterns represented in its labelled
    training data. Unseen services, formats, languages, and failure
    modes can reduce recall or increase false positives.
-   The class-1 score is not calibrated. Probability calibration and
    threshold selection should be performed against a representative
    validation set and the operational cost of false
    positives/negatives.
-   Guardrails cover only explicitly implemented signatures; they are
    not a complete incident-detection rule set.
-   Records are evaluated independently. The system does not correlate
    events over time or across services.
-   Feature hashing can produce collisions and does not preserve an
    interpretable vocabulary.
-   Training happens at process startup. The prototype does not persist
    a fitted model artifact or perform online learning.
-   Output is written to stdout. Alert routing, durable storage,
    dashboards, authentication, and message-broker integration are
    outside this prototype.
-   A memory ceiling does not impose a CPU quota. CPU limits and
    production resource policies require separate load testing.

## References

1\. scikit-learn, [RandomForestClassifier
API](%5Bhttps://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html%5D(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)).
2. scikit-learn, [HashingVectorizer
API](%5Bhttps://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html%5D(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)).
3. scikit-learn, [DictVectorizer
API](%5Bhttps://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html%5D(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)).
4. SciPy, [Sparse matrices
documentation](%5Bhttps://docs.scipy.org/doc/scipy/reference/sparse.html%5D(https://docs.scipy.org/doc/scipy/reference/sparse.html)).
5. Docker, [Resource
constraints](%5Bhttps://docs.docker.com/engine/containers/resource_constraints/%5D(https://docs.docker.com/engine/containers/resource_constraints/)).
6. Docker, [Dockerfile
reference](%5Bhttps://docs.docker.com/reference/dockerfile/%5D(https://docs.docker.com/reference/dockerfile/)).
7. Python, [unittest
documentation](%5Bhttps://docs.python.org/3/library/unittest.html%5D(https://docs.python.org/3/library/unittest.html)).
