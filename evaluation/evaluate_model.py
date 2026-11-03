#!/usr/bin/env python3
"""Evaluate the Random Forest classifier and hybrid guardrails on a held-out synthetic set.

Run from repository root:
    python -m evaluation.evaluate_model --output evaluation/results.json

The report is reproducible for a fixed seed and explicitly identifies the
synthetic corpus. It is not a substitute for labelled service telemetry.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import LogAnomalyDetector  # noqa: E402
from evaluation.dataset import build_corpus  # noqa: E402


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": int(tp), "true_negative": int(tn),
        "false_positive": int(fp), "false_negative": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "accuracy": round((tp + tn) / len(y_true), 4) if y_true else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results.json")
    args = parser.parse_args()

    train_normal, train_anomaly, validation, test, split = build_corpus(seed=args.seed)
    detector = LogAnomalyDetector().fit(train_normal, train_anomaly)

    # Validation set contains only normal records. This reports the empirical
    # model-only false-positive rate without tuning on the held-out test set.
    validation_scores = [detector.model.predict_proba(detector._features([line]))[0][1]
                         for line in validation]
    validation_fp = sum(score >= 0.5 for score in validation_scores)

    y_true = [label for _, label in test]
    model_predictions: list[int] = []
    hybrid_predictions: list[int] = []
    source_counts: Counter[str] = Counter()
    for line, _ in test:
        matrix = detector._features([line])
        model_probability = float(detector.model.predict_proba(matrix)[0][1])
        model_predictions.append(int(model_probability >= 0.5))
        result = detector.detect(line)
        hybrid_predictions.append(int(result.label == "ANOMALY"))
        source_counts[result.source] += 1

    report = {
        "evaluation_type": "deterministic_synthetic_holdout",
        "seed": args.seed,
        "dataset": {
            "training_normal_records": len(train_normal),
            "training_anomaly_records": len(train_anomaly),
            "validation_normal_records": len(validation),
            "test_normal_records": sum(label == 0 for label in y_true),
            "test_anomaly_records": sum(label == 1 for label in y_true),
            "test_template_generated_normal_records": 160,
            "test_template_generated_anomaly_records": 160,
            "test_challenge_normal_records": 10,
            "test_challenge_anomaly_records": 10,
            **split,
        },
        "random_forest_classifier_only": metrics(y_true, model_predictions),
        "hybrid_classifier_plus_guardrails": metrics(y_true, hybrid_predictions),
        "validation_normal_only": {
            "records": len(validation),
            "classifier_false_positives_at_probability_0_5": int(validation_fp),
            "classifier_false_positive_rate_at_probability_0_5": round(validation_fp / len(validation), 4),
        },
        "hybrid_detection_sources": dict(source_counts),
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "All records are synthetic and generated from fixed templates.",
            "The held-out split prevents exact record reuse but does not represent production log diversity.",
            "Metrics are valid only for this generated corpus and must not be generalized to real services.",
            "The validation set is normal-only and is not used to tune the decision threshold.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    report_path = args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
    print(f"\nReport written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
