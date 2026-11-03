import unittest

from evaluation.dataset import build_corpus
from evaluation.evaluate_model import metrics


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_metrics(self):
        result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 0)
        self.assertEqual(result["precision"], 0.6667)
        self.assertEqual(result["recall"], 1.0)

    def test_template_disjoint_split_is_nonempty_and_balanced(self):
        normal, anomaly, validation, test, split = build_corpus(seed=17)
        self.assertEqual(len(normal), 320)
        self.assertEqual(len(anomaly), 320)
        self.assertEqual(len(validation), 120)
        self.assertEqual(sum(label == 0 for _, label in test), 170)
        self.assertEqual(sum(label == 1 for _, label in test), 170)
        self.assertGreater(split["heldout_normal_templates"], 0)
        self.assertGreater(split["heldout_anomaly_templates"], 0)

    def test_dataset_generation_is_deterministic(self):
        first = build_corpus(seed=99)
        second = build_corpus(seed=99)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
