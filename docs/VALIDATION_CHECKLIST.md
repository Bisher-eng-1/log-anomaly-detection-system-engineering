# Final validation checklist

## Local code and model

- [ ] Create and activate a clean Python virtual environment.
- [ ] Install the exact versions in `requirements.txt`.
- [ ] Run `python -m unittest -v` and retain the complete output.
- [ ] Run `python -m evaluation.evaluate_model --output evaluation/results.json`.
- [ ] Inspect the confusion matrix and all false positives/false negatives.
- [ ] Run `python app.py --demo` and `python app.py --input samples/sample.log`.
- [ ] Run `python benchmarks/benchmark.py --records 1000` and retain the JSON output.
- [ ] If using real service data, use separate labelled normal and anomaly files and document where they came from and how labels were reviewed.

## Docker and memory

- [ ] Install Docker Engine/Desktop and verify `docker version` reports both client and server.
- [ ] Run `docker build -t log-anomaly-detection-system .`.
- [ ] Run `python scripts/validate_docker.py`.
- [ ] Confirm `result` is `PASS`, `oom_killed` is `false`, and `memory_limit_mib` is `240.0`.
- [ ] Inspect `observed_peak_mib` and retain `docker-validation-results.json`.
- [ ] Repeat on the target host if the deployment environment differs.

Do not replace missing measurements with estimates. A configured memory cap and a successful build are not substitutes for a measured cgroup peak.
