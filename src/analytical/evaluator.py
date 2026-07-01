from collections import defaultdict
from datetime import datetime
from statistics import mean, median


class Evaluator:
    """Avalia alertas por episódio sem expor o ground truth à inferência."""

    def __init__(self, dataset, config=None):
        self.dataset = dataset
        self.config = config or {}

    def _ground_truth(self):
        occurrences = {}
        timestamps = {}
        physical_states = {}
        for frame in self.dataset.frames():
            metadata = self.dataset.read_json(frame.metadata_path)
            timestamps[frame.frame] = metadata.get("timestamp")
            if not frame.ground_truth_path.exists():
                continue
            payload = self.dataset.read_json(frame.ground_truth_path)
            physical_states[frame.frame] = {
                int(animal_id): state
                for animal_id, state in payload.get(
                    "animal_physical_states", {}
                ).items()
            }
            for occurrence in payload.get("occurrences", []):
                animal_id = occurrence.get("animal_id")
                if animal_id is None:
                    continue
                occurrence_id = occurrence["id"]
                item = occurrences.setdefault(
                    occurrence_id,
                    {
                        "id": occurrence_id,
                        "type": occurrence["type"],
                        "animal_id": int(animal_id),
                        "start_frame": frame.frame,
                        "end_frame": frame.frame,
                        "active_frames": [],
                    },
                )
                item["active_frames"].append(frame.frame)
                item["start_frame"] = min(item["start_frame"], frame.frame)
                item["end_frame"] = max(item["end_frame"], frame.frame)

        persistent_states = {
            "animal_death": "dead",
            "animal_missing": "missing",
        }
        for occurrence in occurrences.values():
            expected_state = persistent_states.get(occurrence["type"])
            if expected_state is None:
                continue
            for frame_number in range(
                occurrence["end_frame"] + 1,
                self.dataset.end_frame + 1,
            ):
                state = physical_states.get(frame_number, {}).get(
                    occurrence["animal_id"]
                )
                if state != expected_state:
                    break
                occurrence["active_frames"].append(frame_number)
                occurrence["end_frame"] = frame_number

        for occurrence in occurrences.values():
            occurrence["started_at"] = timestamps.get(occurrence["start_frame"])
            occurrence["ended_at"] = timestamps.get(occurrence["end_frame"])
        return list(occurrences.values())

    @staticmethod
    def _overlap(alert, occurrence, tolerance):
        return not (
            int(alert["last_frame"]) < occurrence["start_frame"] - tolerance
            or int(alert["opened_frame"]) > occurrence["end_frame"] + tolerance
        )

    def _match(self, alerts, occurrences):
        tolerance = max(
            0,
            int(
                self.config.get("evaluation", {}).get(
                    "matching_tolerance_frames", 0
                )
            ),
        )
        candidates = []
        for alert_index, alert in enumerate(alerts):
            for occurrence_index, occurrence in enumerate(occurrences):
                same_key = (
                    alert["type"] == occurrence["type"]
                    and int(alert["track_id"]) == occurrence["animal_id"]
                )
                if not same_key or not self._overlap(alert, occurrence, tolerance):
                    continue
                overlap = len(
                    set(range(int(alert["opened_frame"]), int(alert["last_frame"]) + 1))
                    & set(occurrence["active_frames"])
                )
                distance = abs(int(alert["opened_frame"]) - occurrence["start_frame"])
                candidates.append((-overlap, distance, alert_index, occurrence_index))

        matched_alerts = set()
        matched_occurrences = set()
        matches = []
        for _, _, alert_index, occurrence_index in sorted(candidates):
            if alert_index in matched_alerts or occurrence_index in matched_occurrences:
                continue
            matched_alerts.add(alert_index)
            matched_occurrences.add(occurrence_index)
            matches.append((alert_index, occurrence_index))
        return matches, matched_alerts, matched_occurrences

    @staticmethod
    def _seconds_between(started_at, detected_at):
        if not started_at or not detected_at:
            return None
        try:
            return (datetime.fromisoformat(detected_at) - datetime.fromisoformat(started_at)).total_seconds()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _delay_summary(values):
        if not values:
            return {"mean": None, "median": None, "maximum": None}
        return {
            "mean": round(mean(values), 3),
            "median": round(median(values), 3),
            "maximum": round(max(values), 3),
        }

    def _coverage(self, observations):
        expected_frames = self.dataset.end_frame - self.dataset.start_frame + 1
        analyzed_frames = {
            int(item["frame"])
            for item in (observations or [])
            if self.dataset.start_frame
            <= int(item["frame"])
            <= self.dataset.end_frame
        }
        tracks = {
            int(detection["track_id"])
            for item in (observations or [])
            for detection in item.get("detections", [])
        }
        maximum_observed_count = max(
            (len(item.get("detections", [])) for item in (observations or [])),
            default=0,
        )
        return {
            "expected_frames": expected_frames,
            "analyzed_frames": len(analyzed_frames),
            "missing_frames": sorted(
                set(range(self.dataset.start_frame, self.dataset.end_frame + 1))
                - analyzed_frames
            ),
            "frame_coverage": round(
                len(analyzed_frames) / expected_frames if expected_frames else 0.0,
                4,
            ),
            "distinct_tracks_observed": len(tracks),
            "maximum_observed_count": maximum_observed_count,
            "inventory_reference_source": "not_provided_to_analytical_agent",
        }

    def evaluate(self, alerts, observations=None):
        occurrences = self._ground_truth()
        matches, matched_alerts, matched_occurrences = self._match(alerts, occurrences)

        occurrence_results = []
        delay_frames = []
        delay_seconds = []
        match_by_occurrence = {
            occurrence_index: alert_index
            for alert_index, occurrence_index in matches
        }
        for occurrence_index, occurrence in enumerate(occurrences):
            alert_index = match_by_occurrence.get(occurrence_index)
            alert = alerts[alert_index] if alert_index is not None else None
            frame_delay = (
                int(alert["opened_frame"]) - occurrence["start_frame"]
                if alert
                else None
            )
            seconds_delay = (
                self._seconds_between(occurrence["started_at"], alert["opened_at"])
                if alert
                else None
            )
            if frame_delay is not None:
                delay_frames.append(frame_delay)
            if seconds_delay is not None:
                delay_seconds.append(seconds_delay)
            occurrence_results.append(
                {
                    **occurrence,
                    "detected": alert is not None,
                    "matched_alert_id": alert["alert_id"] if alert else None,
                    "detection_delay_frames": frame_delay,
                    "detection_delay_seconds": seconds_delay,
                    "maximum_confidence": (
                        alert["maximum_confidence"] if alert else None
                    ),
                }
            )

        true_positives = len(matches)
        false_positives = len(alerts) - true_positives
        false_negatives = len(occurrences) - true_positives
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {
            "dataset_id": self.dataset.dataset_id,
            "evaluation_level": "episode",
            "alert_count": len(alerts),
            "occurrence_count": len(occurrences),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_alert_ids": [
                alert["alert_id"]
                for index, alert in enumerate(alerts)
                if index not in matched_alerts
            ],
            "false_negative_occurrence_ids": [
                occurrence["id"]
                for index, occurrence in enumerate(occurrences)
                if index not in matched_occurrences
            ],
            "detection_delay": {
                "frames": self._delay_summary(delay_frames),
                "seconds": self._delay_summary(delay_seconds),
            },
            "coverage": self._coverage(observations),
            "occurrences": occurrence_results,
        }
