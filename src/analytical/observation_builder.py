import json


class ObservationBuilder:
    """Converte metadados do simulador em observações sem respostas ocultas."""

    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir
        self.observations_dir = output_dir / "observations"
        self.observations_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_detection(detection):
        bbox = detection["bbox_xyxy"]
        return {
            "track_id": detection["id"],
            "category": detection["category"],
            "lot_id": detection.get("lot_id"),
            "confidence": detection["confidence"],
            "bbox_xyxy": bbox,
            "center_xy": [
                round((bbox[0] + bbox[2]) / 2, 2),
                round((bbox[1] + bbox[3]) / 2, 2),
            ],
        }

    def build(self):
        observations = []
        for frame in self.dataset.frames():
            if hasattr(frame, "observation_path"):
                observation = self.dataset.read_json(frame.observation_path)
                observation["observed_count"] = len(
                    observation.get("detections", [])
                )
                output_path = self.observations_dir / f"frame_{frame.frame:04d}.json"
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(observation, file, indent=2, ensure_ascii=False)
                observations.append(observation)
                continue

            metadata = self.dataset.read_json(frame.metadata_path)
            detections = [
                self._sanitize_detection(item)
                for item in metadata.get("detections", [])
            ]
            observation = {
                "dataset_id": self.dataset.dataset_id,
                "frame": metadata["frame"],
                "timestamp": metadata["timestamp"],
                "day": metadata["day"],
                "hour": metadata["hora"],
                "weather": {
                    "temperature_c": metadata["temperatura"],
                    "wind": metadata["vento"],
                },
                "inventory_expected": metadata["total_cattle_expected"],
                "observed_count": len(detections),
                "detections": detections,
                "image_refs": {
                    "processed": str(
                        frame.processed_image_path.relative_to(self.dataset.source_root)
                    ),
                    "raw": str(frame.raw_image_path.relative_to(self.dataset.source_root)),
                },
                "tracking_source": "detector_track_id",
            }
            output_path = self.observations_dir / f"frame_{frame.frame:04d}.json"
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(observation, file, indent=2, ensure_ascii=False)
            observations.append(observation)
        return observations
