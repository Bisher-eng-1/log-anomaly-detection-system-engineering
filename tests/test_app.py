import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

from app import (
    DEMO_LOGS,
    MAX_LINE_CHARS,
    NORMAL_BASELINE,
    LogAnomalyDetector,
    main,
    metadata_features,
    parse_http_status,
    parse_severity,
    stream_records,
)


class ParsingTests(unittest.TestCase):
    def test_severity_normalization(self):
        self.assertEqual(parse_severity("INFO: ready"), "INFO")
        self.assertEqual(parse_severity("WARNING: hot"), "WARN")
        self.assertEqual(parse_severity("unstructured message"), "UNKNOWN")

    def test_http_status_requires_status_context(self):
        self.assertEqual(parse_http_status("ERROR: HTTP 503 upstream unavailable"), 503)
        self.assertEqual(parse_http_status("request completed with status_code=500"), 500)
        self.assertEqual(parse_http_status("response status: 502"), 502)
        for line in (
            "INFO: processed 512 records", "INFO: cache size 512 MB",
            "INFO: user id 503 logged in", "ERROR: 503 records were processed",
        ):
            with self.subTest(line=line):
                self.assertIsNone(parse_http_status(line))

    def test_metadata_features_are_context_aware(self):
        features = metadata_features("ERROR: HTTP 503 CPU timeout")
        self.assertEqual(features["severity_is_error"], 1.0)
        self.assertEqual(features["contains_cpu"], 1.0)
        self.assertEqual(features["contains_timeout"], 1.0)
        self.assertEqual(features["contains_http_error"], 1.0)
        self.assertEqual(metadata_features("INFO: processed 512 records")["contains_http_error"], 0.0)


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = LogAnomalyDetector().fit()

    def test_default_baseline_is_sufficiently_varied(self):
        self.assertGreaterEqual(len(NORMAL_BASELINE), 40)

    def test_operational_metadata_has_normal_and_non_normal_values(self):
        rows = [metadata_features(line) for line in NORMAL_BASELINE]
        for feature in (
            "contains_cpu", "contains_memory", "contains_disk", "contains_timeout",
            "contains_exception", "contains_restart", "contains_auth_failure",
            "contains_network_fault", "contains_http_error",
        ):
            with self.subTest(feature=feature):
                self.assertEqual({row[feature] for row in rows}, {0.0, 1.0})

    def test_fit_rejects_too_small_normal_class(self):
        with self.assertRaises(ValueError):
            LogAnomalyDetector().fit(["INFO: one", "INFO: two"])

    def test_fit_rejects_too_small_anomaly_class(self):
        with self.assertRaises(ValueError):
            LogAnomalyDetector().fit(NORMAL_BASELINE, ["ERROR: one", "ERROR: two"])

    def test_detect_requires_fit(self):
        with self.assertRaises(RuntimeError):
            LogAnomalyDetector().detect("INFO: ready")

    def test_probability_and_signed_margin_are_consistent(self):
        result = self.detector.detect("INFO: health check passed")
        self.assertIn(result.label, {"NORMAL", "ANOMALY"})
        self.assertTrue(math.isfinite(result.decision_value))
        self.assertTrue(math.isfinite(result.model_anomaly_score))
        self.assertGreaterEqual(result.model_anomaly_score, 0.0)
        self.assertLessEqual(result.model_anomaly_score, 1.0)
        self.assertAlmostEqual(result.decision_value, 0.5 - result.model_anomaly_score)

    def test_bundled_normal_baseline_is_classified_as_normal(self):
        for line in NORMAL_BASELINE:
            with self.subTest(line=line):
                result = self.detector.detect(line)
                self.assertEqual(result.label, "NORMAL")
                self.assertEqual(result.source, "random_forest_classifier")

    def test_classifier_detects_anomaly_without_guardrail(self):
        result = self.detector.detect("INFO: memory allocation failure terminated worker w2")
        self.assertEqual(result.label, "ANOMALY")
        self.assertEqual(result.source, "random_forest_classifier")

    def test_guardrail_overrides_classifier_decision(self):
        result = self.detector.detect("WARN: High CPU")
        self.assertEqual((result.label, result.source), ("ANOMALY", "operational_guardrail"))

    def test_guardrail_flags_explicit_failure_signatures(self):
        for line in (
            "FATAL: kernel panic", "ERROR: database corruption detected",
            "ERROR: segmentation fault in worker", "ERROR: out of memory while allocating buffer",
        ):
            with self.subTest(line=line):
                result = self.detector.detect(line)
                self.assertEqual((result.label, result.source), ("ANOMALY", "operational_guardrail"))

    def test_guardrail_detects_high_confidence_failure_even_with_info_prefix(self):
        result = self.detector.detect("INFO: database corruption detected in table accounts")
        self.assertEqual((result.label, result.source), ("ANOMALY", "operational_guardrail"))

    def test_guardrail_flags_explicit_http_5xx(self):
        for line in (
            "ERROR: HTTP 500 Internal Server Error", "WARN: upstream HTTP 503 unavailable",
            "ERROR: status_code=599",
        ):
            with self.subTest(line=line):
                result = self.detector.detect(line)
                self.assertEqual((result.label, result.source), ("ANOMALY", "operational_guardrail"))

    def test_numeric_identifiers_and_counts_do_not_trigger_http_rule(self):
        for line in ("INFO: processed 512 records", "INFO: cache size 512 MB", "INFO: user id 503 logged in"):
            with self.subTest(line=line):
                result = self.detector.detect(line)
                self.assertNotEqual(result.source, "operational_guardrail")

    def test_empty_record_is_ignored(self):
        result = self.detector.detect("   ")
        self.assertEqual(result.label, "IGNORED")
        self.assertEqual(result.source, "input_filter")

    def test_long_record_is_bounded(self):
        result = self.detector.detect("INFO: " + "x" * (MAX_LINE_CHARS * 2))
        self.assertLessEqual(len(result.message), MAX_LINE_CHARS)

    def test_demo_stream_contains_records(self):
        self.assertGreaterEqual(len(DEMO_LOGS), 5)


class StreamAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = LogAnomalyDetector().fit()

    def test_stream_skips_blank_lines_and_emits_results(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stream_records(["INFO: ready\n", "\n", "ERROR: HTTP 500\n"], self.detector)
        rendered = output.getvalue()
        self.assertIn("INFO: ready", rendered)
        self.assertIn("ERROR: HTTP 500", rendered)
        self.assertEqual(rendered.count("\n"), 2)

    def test_stream_caps_oversized_physical_lines(self):
        output = io.StringIO()
        oversized = "INFO: " + "x" * (MAX_LINE_CHARS * 3) + "\n"
        with contextlib.redirect_stdout(output):
            stream_records(io.StringIO(oversized + "INFO: next record\n"), self.detector)
        rendered = output.getvalue()
        self.assertIn("INFO: next record", rendered)
        first_result = rendered.splitlines()[0]
        self.assertLessEqual(len(first_result.split(" | ", 1)[1]), MAX_LINE_CHARS)

    def test_demo_command_runs(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--demo"])
        self.assertEqual(code, 0)
        self.assertIn("operational_guardrail", output.getvalue())

    def test_demo_and_input_arguments_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--demo", "--input", "samples/sample.log"])

    def test_missing_input_file_returns_error_code(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = main(["--input", "/definitely/not/a/real/log-file.log"])
        self.assertEqual(code, 2)
        self.assertIn("Unable to read input file", error.getvalue())


    def test_custom_training_files_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            normal = Path(directory) / "normal.log"
            anomaly = Path(directory) / "anomaly.log"
            normal.write_text("\n".join(NORMAL_BASELINE) + "\n", encoding="utf-8")
            anomaly.write_text("\n".join([
                "FATAL: kernel panic", "ERROR: segmentation fault", "ERROR: database corruption",
                "ERROR: out of memory", "ERROR: connection refused", "ERROR: disk full",
                "ERROR: unrecoverable exception", "CRITICAL: service crashed",
            ]) + "\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--demo", "--normal-training", str(normal), "--anomaly-training", str(anomaly)])
            self.assertEqual(code, 0)
            self.assertIn("operational_guardrail", output.getvalue())

    def test_custom_training_files_must_be_supplied_as_a_pair(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = main(["--demo", "--normal-training", "samples/sample.log"])
        self.assertEqual(code, 2)
        self.assertIn("must be supplied together", error.getvalue())

    def test_file_input_is_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.log"
            path.write_text("INFO: ready\nERROR: HTTP 503\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--input", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("HTTP 503", output.getvalue())


if __name__ == "__main__":
    unittest.main()
