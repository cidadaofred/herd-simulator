from collections import defaultdict


class AlertBuilder:
    """Consolida predições consecutivas em episódios de alerta auditáveis."""

    DEFAULT_PRIORITIES = {
        "animal_fallen": "critical",
        "animal_missing": "critical",
        "animal_death": "critical",
        "animal_parturition": "attention",
    }

    def __init__(self, config):
        alerts_cfg = config.get("alerts", {})
        self.merge_gap_frames = max(0, int(alerts_cfg.get("merge_gap_frames", 1)))
        self.merge_gap_frames_by_type = {
            key: max(0, int(value))
            for key, value in alerts_cfg.get("merge_gap_frames_by_type", {}).items()
        }
        self.priorities = {
            **self.DEFAULT_PRIORITIES,
            **alerts_cfg.get("priorities", {}),
        }

    @staticmethod
    def _new_episode(prediction):
        return {
            "type": prediction["type"],
            "track_id": int(prediction["track_id"]),
            "predictions": [prediction],
        }

    def _episodes(self, predictions):
        grouped = defaultdict(list)
        for prediction in predictions:
            grouped[(prediction["type"], int(prediction["track_id"]))].append(
                prediction
            )

        episodes = []
        for items in grouped.values():
            items.sort(key=lambda item: (int(item["frame"]), item["timestamp"]))
            current = self._new_episode(items[0])
            for prediction in items[1:]:
                previous = current["predictions"][-1]
                gap = int(prediction["frame"]) - int(previous["frame"])
                allowed_gap = self.merge_gap_frames_by_type.get(
                    prediction["type"], self.merge_gap_frames
                )
                if gap <= allowed_gap + 1:
                    current["predictions"].append(prediction)
                else:
                    episodes.append(current)
                    current = self._new_episode(prediction)
            episodes.append(current)
        return sorted(
            episodes,
            key=lambda item: (
                int(item["predictions"][0]["frame"]),
                item["type"],
                item["track_id"],
            ),
        )

    def build(self, predictions):
        if not predictions:
            return [], []

        alerts = []
        history = []
        for index, episode in enumerate(self._episodes(predictions), start=1):
            items = episode["predictions"]
            first = items[0]
            last = items[-1]
            maximum = max(items, key=lambda item: float(item["confidence"]))
            alert_id = f"ALERT_{index:05d}"
            alert = {
                "alert_id": alert_id,
                "type": episode["type"],
                "track_id": episode["track_id"],
                "priority": self.priorities.get(episode["type"], "attention"),
                "status": "closed",
                "opened_frame": int(first["frame"]),
                "opened_at": first["timestamp"],
                "last_frame": int(last["frame"]),
                "last_seen_at": last["timestamp"],
                "closed_frame": int(last["frame"]),
                "closed_at": last["timestamp"],
                "maximum_confidence": float(maximum["confidence"]),
                "prediction_count": len(items),
                "prediction_ids": [item["prediction_id"] for item in items],
                "evidence": maximum["evidence"],
                "narrative": maximum.get("narrative"),
            }
            alerts.append(alert)

            history.append(
                {
                    "alert_id": alert_id,
                    "event": "opened",
                    "frame": int(first["frame"]),
                    "timestamp": first["timestamp"],
                    "confidence": float(first["confidence"]),
                }
            )
            for prediction in items[1:]:
                history.append(
                    {
                        "alert_id": alert_id,
                        "event": "updated",
                        "frame": int(prediction["frame"]),
                        "timestamp": prediction["timestamp"],
                        "confidence": float(prediction["confidence"]),
                    }
                )
            history.append(
                {
                    "alert_id": alert_id,
                    "event": "closed",
                    "frame": int(last["frame"]),
                    "timestamp": last["timestamp"],
                    "confidence": float(last["confidence"]),
                }
            )
        return alerts, history
