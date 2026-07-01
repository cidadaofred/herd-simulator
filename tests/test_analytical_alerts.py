import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.analytical.alert_builder import AlertBuilder
from src.analytical.evaluator import Evaluator


def prediction(frame, confidence=0.7):
    return {
        "prediction_id": f"PRED_{frame:05d}",
        "type": "animal_fallen",
        "track_id": 17,
        "frame": frame,
        "timestamp": f"2026-01-01T{frame:02d}:00:00",
        "confidence": confidence,
        "evidence": {"stationary_window_frames": 4},
        "narrative": "Evidência temporal.",
    }


class FakeDataset:
    dataset_id = "test_dataset"
    start_frame = 1
    end_frame = 5

    def __init__(self, root):
        self.root = Path(root)
        self.items = []
        for frame in range(1, 6):
            metadata_path = self.root / f"metadata_{frame}.json"
            ground_truth_path = self.root / f"ground_truth_{frame}.json"
            metadata_path.write_text(
                json.dumps({"timestamp": f"2026-01-01T{frame:02d}:00:00"}),
                encoding="utf-8",
            )
            occurrences = []
            if 2 <= frame <= 4:
                occurrences.append(
                    {
                        "id": "OCC_001",
                        "type": "animal_fallen",
                        "animal_id": 17,
                    }
                )
            ground_truth_path.write_text(
                json.dumps({"occurrences": occurrences}), encoding="utf-8"
            )
            self.items.append(
                SimpleNamespace(
                    frame=frame,
                    metadata_path=metadata_path,
                    ground_truth_path=ground_truth_path,
                )
            )

    def frames(self):
        yield from self.items

    @staticmethod
    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class AlertBuilderTests(unittest.TestCase):
    def test_groups_contiguous_predictions_and_preserves_maximum(self):
        alerts, history = AlertBuilder(
            {
                "alerts": {
                    "merge_gap_frames": 0,
                    "priorities": {"animal_fallen": "critical"},
                }
            }
        ).build([prediction(3, 0.6), prediction(4, 0.9), prediction(7, 0.8)])

        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["prediction_count"], 2)
        self.assertEqual(alerts[0]["maximum_confidence"], 0.9)
        self.assertEqual(alerts[0]["priority"], "critical")
        self.assertEqual([item["event"] for item in history[:3]], ["opened", "updated", "closed"])


class EvaluatorTests(unittest.TestCase):
    def test_persistent_death_crosses_composed_dataset_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = FakeDataset(directory)
            for item in dataset.items:
                payload = {
                    "occurrences": (
                        [
                            {
                                "id": "DEATH_001",
                                "type": "animal_death",
                                "animal_id": 19,
                            }
                        ]
                        if item.frame == 2
                        else []
                    ),
                    "animal_physical_states": (
                        {"19": "dead"} if item.frame >= 2 else {}
                    ),
                }
                item.ground_truth_path.write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            occurrences = Evaluator(dataset)._ground_truth()

        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0]["start_frame"], 2)
        self.assertEqual(occurrences[0]["end_frame"], 5)
        self.assertEqual(occurrences[0]["active_frames"], [2, 3, 4, 5])

    def test_episode_metrics_delay_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = FakeDataset(directory)
            alerts, _ = AlertBuilder({"alerts": {"merge_gap_frames": 0}}).build(
                [prediction(3), prediction(4)]
            )
            observations = [
                {
                    "frame": frame,
                    "inventory_expected": 1,
                    "detections": [{"track_id": 17}],
                }
                for frame in range(1, 6)
            ]
            result = Evaluator(dataset).evaluate(alerts, observations)

        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["f1_score"], 1.0)
        self.assertEqual(result["detection_delay"]["frames"]["mean"], 1)
        self.assertEqual(result["coverage"]["frame_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
