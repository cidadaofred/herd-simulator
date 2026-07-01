import unittest

from src.analytical.rule_engine import RuleEngine


def detection(track_id, x, y, appearance=None):
    item = {
        "track_id": track_id,
        "confidence": 0.92,
        "center_xy": [x, y],
        "bbox_xyxy": [x - 10, y - 10, x + 10, y + 10],
    }
    if appearance is not None:
        item["appearance"] = {"foreground_rgb": appearance}
    return item


def observation(frame, detections):
    return {
        "frame": frame,
        "timestamp": f"2026-01-{1 + (frame - 1) // 12:02d}T10:00:00",
        "hour": 10,
        "detections": detections,
        "observed_count": len(detections),
    }


def config():
    return {
        "stationary": {
            "window_frames": 4,
            "maximum_span_px": 14,
            "minimum_isolation_px": 45,
            "minimum_confidence": 0.55,
        },
        "missing": {
            "persistence_frames": 4,
            "minimum_observed_herd_ratio": 0.6,
            "analysis_start_hour": 9,
            "analysis_end_hour": 17,
            "minimum_confidence": 0.55,
        },
        "death": {
            "enabled": True,
            "history_frames": 72,
            "minimum_stationary_frames": 48,
            "minimum_observed_frames": 34,
            "maximum_position_span_px": 14,
            "appearance_baseline_samples": 6,
            "appearance_recent_samples": 3,
            "minimum_color_change_rgb": 28,
            "minimum_confidence": 0.7,
        },
        "parturition": {
            "enabled": True,
            "minimum_prior_frames": 12,
            "prepartum_window_frames": 4,
            "maximum_mother_step_px": 40,
            "minimum_isolation_px": 20,
            "new_track_window_frames": 4,
            "maximum_calf_distance_px": 20,
            "minimum_persistence_frames": 3,
            "minimum_confidence": 0.53,
        },
    }


class LifecycleAnalyticalRuleTests(unittest.TestCase):
    def test_adapts_visual_population_without_inventing_a_cause(self):
        settings = config()
        settings["population_adaptation"] = {
            "minimum_detection_confidence": 0.55,
            "new_track_confirmation_frames": 2,
            "retirement_observed_absence_frames": 3,
            "adaptation_start_hour": 9,
            "adaptation_end_hour": 17,
        }
        observations = [
            observation(1, [detection(1, 100, 100), detection(2, 150, 100)]),
            observation(2, [detection(1, 101, 100), detection(2, 151, 100)]),
            observation(3, [detection(1, 102, 100), detection(2, 152, 100), detection(3, 200, 100)]),
            observation(4, [detection(1, 103, 100), detection(2, 153, 100), detection(3, 201, 100)]),
            observation(5, [detection(1, 104, 100), detection(3, 202, 100)]),
            observation(6, [detection(1, 105, 100), detection(3, 203, 100)]),
            observation(7, [detection(1, 106, 100), detection(3, 204, 100)]),
        ]

        engine = RuleEngine(settings)
        engine.analyze(observations)

        self.assertEqual(engine.reference_tracks, {1, 3})
        self.assertEqual(observations[-1]["population_estimate"]["count"], 2)
        self.assertEqual(len(engine.population_history), 3)
        self.assertTrue(
            all(item["cause"] == "unknown_not_inferred" for item in engine.population_history)
        )

    def test_detects_parturition_from_new_persistent_nearby_track(self):
        observations = []
        for frame in range(1, 19):
            mother_x = 100 + frame
            detections = [
                detection(1, mother_x, 100),
                detection(2, 260, 250),
            ]
            if frame >= 15:
                detections.append(detection(3, mother_x + 9, 106))
            observations.append(observation(frame, detections))

        predictions = RuleEngine(config()).analyze(observations)
        births = [item for item in predictions if item["type"] == "animal_parturition"]

        self.assertEqual(len(births), 1)
        self.assertEqual(births[0]["track_id"], 1)
        self.assertEqual(births[0]["related_track_id"], 3)

    def test_detects_death_after_long_immobility_and_color_change(self):
        observations = []
        for frame in range(1, 61):
            appearance = [220, 220, 210] if frame < 56 else [115, 105, 80]
            observations.append(
                observation(frame, [detection(5, 100, 100, appearance)])
            )

        predictions = RuleEngine(config()).analyze(observations)
        deaths = [item for item in predictions if item["type"] == "animal_death"]

        self.assertEqual(len(deaths), 1)
        self.assertEqual(deaths[0]["track_id"], 5)
        self.assertGreater(deaths[0]["evidence"]["appearance_change_rgb"], 28)
        self.assertFalse(
            any(
                item["type"] == "animal_fallen" and item["track_id"] == 5
                for item in predictions
            )
        )


if __name__ == "__main__":
    unittest.main()
