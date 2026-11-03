#!/usr/bin/env python3
"""Streaming log anomaly detection using sparse features and a Random Forest classifier."""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from training_data import (
    BASELINE_NORMALS as NORMAL_BASELINE,
    DEFAULT_ANOMALY_TRAINING,
    build_training_data,
)

SEVERITY_RE = re.compile(
    r"^\s*(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\s*:", re.IGNORECASE
)
HTTP_STATUS_RE = re.compile(
    r"(?:\bHTTP(?:/\d(?:\.\d)?)?\s+"
    r"|\bstatus(?:_code)?\s*[=: ]\s*"
    r"|\bresponse\s+status\s*[=: ]\s*)"
    r"([1-5]\d\d)\b",
    re.IGNORECASE,
)
MAX_LINE_CHARS = 16_384

# Demo stream used by the CLI and repeatable microbenchmark.
DEMO_LOGS = [
    "INFO: System running",
    "INFO: health check passed",
    "WARN: High CPU",
    "INFO: request completed with HTTP 200",
    "ERROR: HTTP 500 Internal Server Error",
    "INFO: worker is ready",
]


@dataclass(frozen=True)
class Detection:
    """Immutable result for one input record."""

    label: str
    decision_value: float
    model_anomaly_score: float
    message: str
    source: str


def parse_severity(line: str) -> str:
    """Return a normalized severity, or UNKNOWN when no prefix is present."""
    match = SEVERITY_RE.match(line)
    if not match:
        return "UNKNOWN"
    severity = match.group(1).upper()
    return "WARN" if severity == "WARNING" else severity


def parse_http_status(line: str) -> int | None:
    """Extract an HTTP status only when it appears in an explicit status context."""
    match = HTTP_STATUS_RE.search(line)
    return int(match.group(1)) if match else None


def metadata_features(line: str) -> dict[str, float | str]:
    """Build interpretable metadata features for a single log record."""
    severity = parse_severity(line)
    lower = line.lower()
    status = parse_http_status(line)
    indicators = {
        "contains_http_error": status is not None and status >= 400,
        "contains_exception": any(
            token in lower for token in ("exception", "traceback", "stack trace")
        ),
        "contains_timeout": "timeout" in lower or "timed out" in lower,
        "contains_cpu": "cpu" in lower,
        "contains_memory": "memory" in lower or "oom" in lower,
        "contains_disk": "disk" in lower or "storage" in lower,
        "contains_auth_failure": any(
            token in lower for token in ("unauthorized", "forbidden", "authentication failed")
        ),
        "contains_restart": "restart" in lower or "crash" in lower,
        "contains_network_fault": any(
            token in lower for token in ("connection refused", "network", "unreachable")
        ),
    }
    features: dict[str, float | str] = {
        "length_bucket": str(min(len(line) // 40, 5)),
        "severity_is_debug": float(severity == "DEBUG"),
        "severity_is_info": float(severity == "INFO"),
        "severity_is_warn": float(severity == "WARN"),
        "severity_is_error": float(severity == "ERROR"),
        "severity_is_critical": float(severity in {"CRITICAL", "FATAL"}),
        "severity_is_unknown": float(severity == "UNKNOWN"),
    }
    features.update({key: float(value) for key, value in indicators.items()})
    return features


class LogAnomalyDetector:
    """Supervised Random Forest detector with explicit operational guardrails."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.text_vectorizer = HashingVectorizer(
            n_features=2**10,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.metadata_vectorizer = DictVectorizer(sparse=True)
        self.model = RandomForestClassifier(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=self.random_state,
        )
        self._fitted = False

    @staticmethod
    def _clean(line: str) -> str:
        """Trim whitespace and cap a record to a predictable maximum size."""
        return line.strip()[:MAX_LINE_CHARS]

    def _features(self, lines: list[str]):
        text_matrix = self.text_vectorizer.transform(lines)
        metadata_matrix = self.metadata_vectorizer.transform(
            [metadata_features(line) for line in lines]
        )
        return hstack((text_matrix, metadata_matrix), format="csr")

    def fit(
        self,
        baseline: Iterable[str] | None = None,
        anomalies: Iterable[str] | None = None,
    ) -> "LogAnomalyDetector":
        """Train a supervised classifier using labelled normal and anomaly logs.

        The bundled default corpus is synthetic and intended for a reproducible
        prototype. A real deployment must provide reviewed, labelled service
        logs for the classifier and representative normal logs for calibration.
        """
        if baseline is None and anomalies is None:
            records, labels = build_training_data(seed=self.random_state)
        else:
            normal_source = NORMAL_BASELINE if baseline is None else baseline
            anomaly_source = DEFAULT_ANOMALY_TRAINING if anomalies is None else anomalies
            normal_lines = [self._clean(line) for line in normal_source]
            anomaly_lines = [self._clean(line) for line in anomaly_source]
            records = [line for line in normal_lines + anomaly_lines if line]
            labels = ([0] * len([line for line in normal_lines if line])
                      + [1] * len([line for line in anomaly_lines if line]))
        normal_count = sum(label == 0 for label in labels)
        anomaly_count = sum(label == 1 for label in labels)
        if normal_count < 8 or anomaly_count < 8:
            raise ValueError("Training requires at least 8 normal and 8 anomaly records.")
        matrix = self._features_for_fit(records)
        self.model.fit(matrix, labels)
        self._fitted = True
        return self

    def _features_for_fit(self, lines: list[str]):
        text_matrix = self.text_vectorizer.transform(lines)
        metadata = [metadata_features(line) for line in lines]
        metadata_matrix = self.metadata_vectorizer.fit_transform(metadata)
        return hstack((text_matrix, metadata_matrix), format="csr")

    def detect(self, line: str) -> Detection:
        """Classify one record and expose the ML probability and rule source."""
        if not self._fitted:
            raise RuntimeError("Call fit() before detect().")
        message = self._clean(line)
        if not message:
            return Detection("IGNORED", 0.0, 0.0, message, "input_filter")

        matrix = self._features([message])
        probabilities = self.model.predict_proba(matrix)[0]
        class_index = list(self.model.classes_).index(1)
        anomaly_probability = float(probabilities[class_index])
        if not math.isfinite(anomaly_probability):
            raise ArithmeticError("Classifier returned a non-finite anomaly probability.")
        status = parse_http_status(message)
        severity = parse_severity(message)
        lower = message.lower()
        is_server_error = status is not None and 500 <= status <= 599
        is_resource_pressure = severity == "WARN" and any(
            phrase in lower
            for phrase in ("high cpu", "high memory", "memory exhausted", "disk full", "resource exhausted")
        )
        failure_signatures = (
            "exception", "traceback", "kernel panic", "segmentation fault",
            "out of memory", "oom", "database corruption", "disk full",
            "connection refused", "authentication failed", "unrecoverable",
        )
        high_confidence_failure_signatures = (
            "kernel panic", "segmentation fault", "out of memory", "oom",
            "database corruption", "disk full", "connection refused", "unrecoverable",
        )
        is_explicit_failure = (
            severity in {"CRITICAL", "FATAL"}
            or (severity == "ERROR" and any(token in lower for token in failure_signatures))
            or any(token in lower for token in high_confidence_failure_signatures)
        )

        if is_server_error or is_resource_pressure or is_explicit_failure:
            label, source = "ANOMALY", "operational_guardrail"
        elif anomaly_probability >= 0.5:
            label, source = "ANOMALY", "random_forest_classifier"
        else:
            label, source = "NORMAL", "random_forest_classifier"

        # decision_value remains a signed classifier margin for easy inspection:
        # negative means the model probability is above the anomaly threshold.
        decision_value = 0.5 - anomaly_probability
        return Detection(label, decision_value, anomaly_probability, message, source)


def _iter_bounded_lines(source: Iterable[str] | TextIO) -> Iterable[str]:
    """Yield bounded records; use sized reads for file-like input streams."""
    readline = getattr(source, "readline", None)
    if callable(readline):
        while True:
            raw_line = readline(MAX_LINE_CHARS + 2)
            if not raw_line:
                return
            record = raw_line[:MAX_LINE_CHARS]
            if len(raw_line) > MAX_LINE_CHARS and not raw_line.endswith("\n"):
                # Discard the remainder of an oversized physical line without
                # buffering it, so the next yielded record starts at a boundary.
                remainder = raw_line
                while remainder and not remainder.endswith("\n"):
                    remainder = readline(MAX_LINE_CHARS + 2)
            yield record
        return

    # Generic iterables are useful for tests and in-process callers. Their
    # strings already exist in memory, but feature extraction remains bounded.
    for raw_line in source:
        yield raw_line[:MAX_LINE_CHARS]


def stream_records(source: Iterable[str] | TextIO, detector: LogAnomalyDetector) -> None:
    """Process a text stream incrementally without retaining prior records."""
    for raw_line in _iter_bounded_lines(source):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        result = detector.detect(line)
        print(
            f"[{result.label}] decision={result.decision_value:+.6f} "
            f"model_anomaly_score={result.model_anomaly_score:.6f} "
            f"source={result.source} | {result.message}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream log records through a fitted Random Forest anomaly classifier."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--demo", action="store_true", help="process the built-in demonstration stream"
    )
    source_group.add_argument(
        "--input", metavar="PATH", help="read records from a UTF-8 text file"
    )
    parser.add_argument("--normal-training", metavar="PATH", help="labelled normal training log file")
    parser.add_argument("--anomaly-training", metavar="PATH", help="labelled anomaly training log file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.normal_training) != bool(args.anomaly_training):
        print("Both --normal-training and --anomaly-training must be supplied together.", file=sys.stderr)
        return 2

    detector = LogAnomalyDetector()
    try:
        if args.normal_training and args.anomaly_training:
            with Path(args.normal_training).open("r", encoding="utf-8", errors="replace") as normal_file:
                normal_records = list(_iter_bounded_lines(normal_file))
            with Path(args.anomaly_training).open("r", encoding="utf-8", errors="replace") as anomaly_file:
                anomaly_records = list(_iter_bounded_lines(anomaly_file))
            detector.fit(normal_records, anomaly_records)
        else:
            detector.fit()
    except (OSError, ValueError) as exc:
        print(f"Unable to initialise detector: {exc}", file=sys.stderr)
        return 2

    if args.demo:
        stream_records(DEMO_LOGS, detector)
        return 0

    if args.input:
        try:
            with Path(args.input).open("r", encoding="utf-8", errors="replace") as source:
                stream_records(source, detector)
        except OSError as exc:
            print(f"Unable to read input file: {exc}", file=sys.stderr)
            return 2
    else:
        stream_records(sys.stdin, detector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
