# Log Anomaly Detector

A streaming log anomaly classifier built to run cleanly inside a strict **240 MiB** Docker memory cap using sparse hashing features, single-threaded scikit-learn, and optional regex/keyword guardrails.

## Design choices

- **Bounded memory (`240 MiB` cap):** `HashingVectorizer(n_features=1024)` replaces an unbounded vocabulary dictionary, `n_jobs=1` prevents scikit-learn from forking worker copies across cores, and `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` prevent BLAS thread arenas from inflating RSS.
- **Bounded streaming & batch inference:** Input lines are capped at `16,384` chars without buffering full oversized lines, custom training inputs are bounded (`--max-train-records`), and `detect_batch()` vectorizes chunks of `64` lines per `predict_proba` call instead of running single-row sparse allocations.
- **ML vs. hybrid mode:** Explicit failure prefixes (`FATAL`, `CRITICAL`, HTTP `5xx`, `kernel panic`, `oom`, `segmentation fault`) are caught deterministically by guardrails unless `--mode ml-only` is passed, leaving the Random Forest to score subtler log patterns.

## Run locally

```bash
pip install -r requirements.txt
python app.py --demo
python app.py --demo --mode ml-only --format json
python app.py --input samples/sample.log --format json
```

Stream from stdin:

```bash
cat samples/sample.log | python app.py --batch-size 64
```

Train on your own labelled logs:

```bash
python app.py --normal-training normal.log --anomaly-training anomaly.log --input app.log
```

## Docker (240 MiB cap)

```bash
docker compose up --build
python scripts/validate_docker.py
```

The compose service runs as non-root (`10001:10001`), mounts the root filesystem read-only, drops all Linux capabilities, and enforces `mem_limit: 240m` / `memswap_limit: 240m`.

## Evaluation & tests

```bash
python -m unittest -v
python -m evaluation.evaluate_model --output evaluation/results.json
python benchmarks/benchmark.py --records 5000
```

On the bundled template-disjoint synthetic holdout (`340` test records across unseen templates and prefix-free challenge cases):
- **Random Forest only (`ml-only`):** Precision `0.9843`, Recall `0.7353`, F1 `0.8418`, FPR `0.0118`
- **Hybrid (RF + guardrails):** Precision `0.9877`, Recall `0.9471`, F1 `0.9670`, FPR `0.0118`

## Limitations

- **Synthetic training corpus:** Bundled training and evaluation logs are generated from templates. For real production logs, supply `--normal-training` and `--anomaly-training` or retrain on labelled telemetry.
- **Fixed threshold (`0.5`):** Classifier probabilities are not calibrated for severe class imbalance.
- **Stateless per line:** Records are classified independently, so multi-line stack traces and cross-request rate spikes are not aggregated across windows.
