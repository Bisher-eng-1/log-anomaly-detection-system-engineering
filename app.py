#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO

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
    r"^\s*(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\s*:",
    re.IGNORECASE,
)
HTTP_STATUS_RE = re.compile(
    r"(?:\bHTTP(?:/\d(?:\.\d)?)?\s+"
    r"|\bstatus(?:_code)?\s*[=: ]\s*"
    r"|\bresponse\s+status\s*[=: ]\s*)"
    r"([1-5]\d\d)\b",
    re.IGNORECASE,
)
MAX_LINE_CHARS = 16_384
MAX_TRAINING_RECORDS = 50_000
DEFAULT_BATCH_SIZE = 64

DEMO_LOGS = [
    "INFO: System running",
    "INFO: health check passed",
    "WARN: High CPU",
    "INFO: request completed with HTTP 200",
    "ERROR: HTTP 500 Internal Server Error",
    "INFO: worker is ready",
]

FAILURE_SIGNATURES = (
    "exception",
    "traceback",
    "kernel panic",
    "segmentation fault",
    "out of memory",
    "oom",
    "database corruption",
    "disk full",
    "connection refused",
    "authentication failed",
    "unrecoverable",
)
HIGH_CONFIDENCE_SIGNATURES = (
    "kernel panic",
    "segmentation fault",
    "out of memory",
    "oom",
    "database corruption",
    "disk full",
    "connection refused",
    "unrecoverable",
)
RESOURCE_PRESSURE_PHRASES = (
    "high cpu",
    "high memory",
    "memory exhausted",
    "disk full",
    "resource exhausted",
)


@dataclass(frozen=True)
class Detection:
    label: str
    decision_value: float
    model_anomaly_score: float
    message: str
    source: str


def parse_severity(line: str) -> str:
    match = SEVERITY_RE.match(line)
    if not match:
        return "UNKNOWN"
    severity = match.group(1).upper()
    return "WARN" if severity == "WARNING" else severity


def parse_http_status(line: str) -> int | None:
    match = HTTP_STATUS_RE.search(line)
    return int(match.group(1)) if match else None


def metadata_features(line: str) -> dict[str, float | str]:
    severity = parse_severity(line)
    lower = line.lower()
    status = parse_http_status(line)
    indicators = {
        "contains_http_error": status is not None and status >= 400,
        "contains_exception": any(t in lower for t in ("exception", "traceback", "stack trace")),
        "contains_timeout": "timeout" in lower or "timed out" in lower,
        "contains_cpu": "cpu" in lower,
        "contains_memory": "memory" in lower or "oom" in lower,
        "contains_disk": "disk" in lower or "storage" in lower,
        "contains_auth_failure": any(
            t in lower for t in ("unauthorized", "forbidden", "authentication failed")
        ),
        "contains_restart": "restart" in lower or "crash" in lower,
        "contains_network_fault": any(
            t in lower for t in ("connection refused", "network", "unreachable")
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
    features.update({k: float(v) for k, v in indicators.items()})
    return features


def _matches_guardrail(message: str) -> bool:
    status = parse_http_status(message)
    if status is not None and 500 <= status <= 599:
        return True
    severity = parse_severity(message)
    lower = message.lower()
    if severity == "WARN" and any(p in lower for p in RESOURCE_PRESSURE_PHRASES):
        return True
    return (
        severity in {"CRITICAL", "FATAL"}
        or (severity == "ERROR" and any(t in lower for t in FAILURE_SIGNATURES))
        or any(t in lower for t in HIGH_CONFIDENCE_SIGNATURES)
    )


class LogAnomalyDetector:
    def __init__(self, random_state: int = 42, use_guardrails: bool = True) -> None:
        self.random_state = random_state
        self.use_guardrails = use_guardrails
        # Fixed hash dimension + n_jobs=1 avoids vocabulary growth and process-fork RSS spikes.
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
        return line.strip()[:MAX_LINE_CHARS]

    def _features(self, lines: list[str]):
        text_matrix = self.text_vectorizer.transform(lines)
        metadata_matrix = self.metadata_vectorizer.transform(
            [metadata_features(line) for line in lines]
        )
        return hstack((text_matrix, metadata_matrix), format="csr")

    def _features_for_fit(self, lines: list[str]):
        text_matrix = self.text_vectorizer.transform(lines)
        metadata_matrix = self.metadata_vectorizer.fit_transform(
            [metadata_features(line) for line in lines]
        )
        return hstack((text_matrix, metadata_matrix), format="csr")

    def fit(
        self,
        baseline: Iterable[str] | None = None,
        anomalies: Iterable[str] | None = None,
    ) -> "LogAnomalyDetector":
        if baseline is None and anomalies is None:
            records, labels = build_training_data(seed=self.random_state)
        else:
            normal_source = NORMAL_BASELINE if baseline is None else baseline
            anomaly_source = DEFAULT_ANOMALY_TRAINING if anomalies is None else anomalies
            normal_lines = [c for c in (self._clean(x) for x in normal_source) if c]
            anomaly_lines = [c for c in (self._clean(x) for x in anomaly_source) if c]
            records = normal_lines + anomaly_lines
            labels = [0] * len(normal_lines) + [1] * len(anomaly_lines)

        normal_count = sum(label == 0 for label in labels)
        anomaly_count = sum(label == 1 for label in labels)
        if normal_count < 8 or anomaly_count < 8:
            raise ValueError("Training requires at least 8 normal and 8 anomaly records.")

        matrix = self._features_for_fit(records)
        self.model.fit(matrix, labels)
        self._fitted = True
        return self

    def detect_batch(self, lines: Sequence[str]) -> list[Detection]:
        if not self._fitted:
            raise RuntimeError("Call fit() before detect().")
        if not lines:
            return []

        cleaned = [self._clean(line) for line in lines]
        non_empty_idx = [i for i, msg in enumerate(cleaned) if msg]
        results: list[Detection | None] = [None] * len(cleaned)

        for i, msg in enumerate(cleaned):
            if not msg:
                results[i] = Detection("IGNORED", 0.0, 0.0, msg, "input_filter")

        if non_empty_idx:
            active_messages = [cleaned[i] for i in non_empty_idx]
            matrix = self._features(active_messages)
            probs = self.model.predict_proba(matrix)
            class_index = list(self.model.classes_).index(1)

            for idx, prob_row in zip(non_empty_idx, probs):
                msg = cleaned[idx]
                anomaly_prob = float(prob_row[class_index])
                if not math.isfinite(anomaly_prob):
                    raise ArithmeticError("Classifier returned a non-finite anomaly probability.")

                if self.use_guardrails and _matches_guardrail(msg):
                    label, source = "ANOMALY", "operational_guardrail"
                elif anomaly_prob >= 0.5:
                    label, source = "ANOMALY", "random_forest_classifier"
                else:
                    label, source = "NORMAL", "random_forest_classifier"

                results[idx] = Detection(label, 0.5 - anomaly_prob, anomaly_prob, msg, source)

        return [item for item in results if item is not None]

    def detect(self, line: str) -> Detection:
        return self.detect_batch([line])[0]


def _iter_bounded_lines(source: Iterable[str] | TextIO) -> Iterable[str]:
    readline = getattr(source, "readline", None)
    if callable(readline):
        while True:
            raw_line = readline(MAX_LINE_CHARS + 2)
            if not raw_line:
                return
            record = raw_line[:MAX_LINE_CHARS]
            if len(raw_line) > MAX_LINE_CHARS and not raw_line.endswith("\n"):
                remainder = raw_line
                while remainder and not remainder.endswith("\n"):
                    remainder = readline(MAX_LINE_CHARS + 2)
            yield record
        return

    for raw_line in source:
        yield raw_line[:MAX_LINE_CHARS]


def _emit_detection(result: Detection, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(result), separators=(",", ":")), flush=True)
    else:
        print(
            f"[{result.label}] decision={result.decision_value:+.6f} "
            f"model_anomaly_score={result.model_anomaly_score:.6f} "
            f"source={result.source} | {result.message}",
            flush=True,
        )


def stream_records(
    source: Iterable[str] | TextIO,
    detector: LogAnomalyDetector,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_format: str = "text",
) -> None:
    chunk: list[str] = []
    effective_batch = max(1, batch_size)
    for raw_line in _iter_bounded_lines(source):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        chunk.append(line)
        if len(chunk) >= effective_batch:
            for result in detector.detect_batch(chunk):
                _emit_detection(result, output_format)
            chunk.clear()

    if chunk:
        for result in detector.detect_batch(chunk):
            _emit_detection(result, output_format)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream log records through a Random Forest anomaly detector."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--demo", action="store_true")
    source_group.add_argument("--input", metavar="PATH")
    parser.add_argument("--normal-training", metavar="PATH")
    parser.add_argument("--anomaly-training", metavar="PATH")
    parser.add_argument("--mode", choices=("hybrid", "ml-only"), default="hybrid")
    parser.add_argument("--format", dest="output_format", choices=("text", "json"), default="text")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-train-records", type=int, default=MAX_TRAINING_RECORDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.normal_training) != bool(args.anomaly_training):
        print("Both --normal-training and --anomaly-training must be supplied together.", file=sys.stderr)
        return 2

    detector = LogAnomalyDetector(use_guardrails=(args.mode == "hybrid"))
    try:
        if args.normal_training and args.anomaly_training:
            cap = max(8, args.max_train_records)
            with Path(args.normal_training).open("r", encoding="utf-8", errors="replace") as nf:
                normal_records = list(itertools.islice(_iter_bounded_lines(nf), cap))
            with Path(args.anomaly_training).open("r", encoding="utf-8", errors="replace") as af:
                anomaly_records = list(itertools.islice(_iter_bounded_lines(af), cap))
            detector.fit(normal_records, anomaly_records)
        else:
            detector.fit()
    except (OSError, ValueError) as exc:
        print(f"Unable to initialise detector: {exc}", file=sys.stderr)
        return 2

    if args.demo:
        stream_records(
            DEMO_LOGS,
            detector,
            batch_size=args.batch_size,
            output_format=args.output_format,
        )
        return 0

    if args.input:
        try:
            with Path(args.input).open("r", encoding="utf-8", errors="replace") as source:
                stream_records(
                    source,
                    detector,
                    batch_size=args.batch_size,
                    output_format=args.output_format,
                )
        except OSError as exc:
            print(f"Unable to read input file: {exc}", file=sys.stderr)
            return 2
    else:
        stream_records(
            sys.stdin,
            detector,
            batch_size=args.batch_size,
            output_format=args.output_format,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
