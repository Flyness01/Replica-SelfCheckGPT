import unittest

from analyze_results import summarize


class SummarizeTests(unittest.TestCase):
    def test_summarizes_repeated_runs(self):
        report = {
            "samples": [
                {"prompt_tps": 10.0, "generation_tps": 20.0, "peak_memory_gb": 1.0, "text": "same"},
                {"prompt_tps": 30.0, "generation_tps": 22.0, "peak_memory_gb": 1.2, "text": "same"},
                {"prompt_tps": 50.0, "generation_tps": 24.0, "peak_memory_gb": 1.1, "text": "same"},
            ]
        }
        summary = summarize(report)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["mean_generation_tps"], 22.0)
        self.assertEqual(summary["mean_warm_prompt_tps"], 40.0)
        self.assertEqual(summary["warm_to_cold_prompt_ratio"], 4.0)
        self.assertTrue(summary["identical_outputs"])
        self.assertEqual(summary["peak_memory_gb"], 1.2)

    def test_rejects_empty_report(self):
        with self.assertRaises(ValueError):
            summarize({"samples": []})


if __name__ == "__main__":
    unittest.main()
