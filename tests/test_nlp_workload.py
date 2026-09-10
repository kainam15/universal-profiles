import unittest

from acprof.workloads.nlp import NLPWorkloadGenerator


class NLPWorkloadCompatibilityTests(unittest.TestCase):
    def test_zero_shot_has_nonempty_deterministic_labels_and_batch(self):
        generator = NLPWorkloadGenerator("model", "zero-shot-classification", 3)
        first = generator.generate(8)
        self.assertEqual(first, generator.generate(8))
        self.assertGreaterEqual(len(first["candidate_labels"]), 2)
        self.assertEqual(first["batch_size"], 3)

    def test_pair_tasks_scale_candidate_length_with_fixed_query_and_count(self):
        for task in ("sentence-similarity", "text-ranking"):
            generator = NLPWorkloadGenerator("model", task, 2)
            small, large = generator.generate(8), generator.generate(16)
            self.assertEqual(small["query"], large["query"])
            self.assertEqual(len(small["documents"]), 2)
            self.assertEqual([len(x.split()) for x in small["documents"]], [8, 8])
            self.assertEqual([len(x.split()) for x in large["documents"]], [16, 16])
            self.assertEqual(small["batch_size"], 2)

    def test_table_qa_scales_real_rows_and_reports_row_defaults(self):
        generator = NLPWorkloadGenerator("model", "table-question-answering", 2)
        payload = generator.generate(8)
        self.assertEqual(payload["input_scale_type"], "table_rows")
        self.assertEqual({len(col) for col in payload["table"].values()}, {8})
        self.assertTrue(all(isinstance(v, str) for col in payload["table"].values() for v in col))
        self.assertEqual(generator.scale_label(8), "rows8")
        self.assertEqual(generator.default_input_scales(), [1, 2, 4, 8, 16, 32])

    def test_table_qa_rejects_fractional_or_empty_row_counts(self):
        generator = NLPWorkloadGenerator("model", "table-question-answering", 1)
        for scale in (0, -1, 1.5):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "positive integer"):
                generator.generate(scale)


if __name__ == "__main__":
    unittest.main()
