# Technical walkthrough

## 1. Runtime lifecycle

1. `main()` parses the input mode and optional labelled training-file paths.
2. `LogAnomalyDetector.fit()` builds a sparse training matrix and fits the Random Forest classifier.
3. `stream_records()` consumes standard input, a file, or the built-in demo one record at a time.
4. `detect()` normalises the record, extracts text and metadata features, obtains the class-1 probability estimate, and evaluates operational guardrails.
5. The returned immutable `Detection` object records the final label, signed margin, model score, original bounded message, and decision source.

## 2. Why sparse feature engineering

The text vectorizer has a fixed feature width and does not build a vocabulary that grows with incoming records. `DictVectorizer` creates a compact representation of explicitly named operational indicators. `scipy.sparse.hstack(..., format="csr")` combines the two without converting the mostly-zero text matrix to a dense array.

This controls feature-matrix growth, but it does not make memory usage constant in every respect. Model training, Python objects, dependency imports, and classifier trees also consume memory. That is why the Docker cgroup measurement is a separate validation step.

## 3. Why a supervised classifier plus guardrails

The classifier learns patterns from labelled normal and anomaly records. This is appropriate for a prototype where example labels are available. The operational rules handle a small set of high-confidence events whose meaning is explicit in the log format, such as a contextual HTTP 5xx response or a fatal severity prefix.

The source field makes the distinction auditable:

- `random_forest_classifier`: the model probability crossed the configured 0.50 threshold.
- `operational_guardrail`: a deterministic operational rule forced the anomaly label, including high-confidence failure phrases that can appear with an inconsistent severity prefix.
- `input_filter`: the record was empty and was not scored.

A high model score is not a calibrated incident probability. Threshold selection should be validated against service-specific data and the cost of missed incidents versus noisy alerts.

## 4. Evaluation protocol

`evaluation/dataset.py` separates training and test message templates. The test set also includes challenge records with benign mentions of terms such as `exception`, CPU, and numeric identifiers, plus failure descriptions without a conventional `ERROR` prefix. This makes the benchmark less trivial than replaying the training examples.

`evaluation/evaluate_model.py` computes confusion-matrix counts and derives precision, recall, F1, false-positive rate, and accuracy. The report keeps classifier-only and hybrid results separate so the contribution of the guardrails is visible.

Even with template-disjoint splits, the corpus is synthetic and limited. A production-quality evaluation needs real, reviewed, labelled records, temporal/service-level separation, representative class prevalence, and a documented annotation process.

## 5. Docker memory validation

The Docker configuration requests a 240 MiB memory limit and an equal memory-plus-swap limit. `scripts/validate_docker.py` checks that the daemon accepted this exact byte limit, runs the demo, then starts a probe that trains the model and performs 500 inference calls. It records the container's cgroup high-water mark when available and also samples `docker stats`.

A process RSS measurement from a normal host run is not interchangeable with a container cgroup peak. A successful local unit-test run does not establish that the Docker workload fits the configured memory ceiling. The validation script therefore reports a pass only when it has a real observed memory value, a successful probe exit, no OOM kill, and the expected configured limit.

## 6. Important engineering trade-offs

- **Hashing collisions:** fixed-width feature hashing can map distinct tokens to the same feature. The trade-off is bounded dimensionality and no vocabulary fit step.
- **Template dependence:** the classifier can learn the training corpus' wording. Domain-specific training and held-out evaluation are essential.
- **Independent records:** there is no sequence model or cross-service correlation. A multi-line stack trace is treated as separate records.
- **Startup training:** the model is trained on process startup. This simplifies the prototype but increases startup time and does not provide model versioning or hot reload.
- **Guardrail scope:** explicit rules improve recall for known signatures but cannot replace a complete operational policy.
- **Resource constraints:** the 240 MiB setting is a configured ceiling. The measured peak must be captured on the target Docker runtime.

## 7. Suggested committee discussion topics

1. Why use a Random Forest classifier for this labelled prototype instead of an unsupervised detector?
2. What does `predict_proba` mean here, and why is it not automatically a calibrated operational risk score?
3. Why use hashing and sparse matrices for log text?
4. How do you distinguish a real HTTP status from an ordinary numeric identifier?
5. What does the source field tell an operator about the final decision?
6. Why can synthetic test metrics not be generalized to production traffic?
7. Why is a Docker memory limit different from a measured memory footprint?
8. What would change before deploying this system against a real multi-service environment?
