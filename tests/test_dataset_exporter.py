import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.analytical.dataset_exporter import DatasetSnapshotExporter


class DatasetSnapshotExporterTest(unittest.TestCase):
    def _write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _source(self, root):
        source = root / "allowed" / "source"
        metadata = {
            "frame": 1,
            "timestamp": "2026-06-20T06:00:00",
            "day": 1,
            "hora": 6,
            "temperatura": 10,
            "vento": 4,
            "total_cattle_expected": 2,
            "spatial_mode": "hidden_mode",
            "cattle_positions": [{"id": 17, "physical_state": "fallen"}],
            "administrative_events": [{"type": "animal_sale"}],
            "detections": [
                {
                    "id": 17,
                    "category": "vaca_lactante",
                    "lot_id": "secret_lot",
                    "confidence": 0.91,
                    "bbox_xyxy": [10, 20, 30, 40],
                    "visibility_status": "partial",
                }
            ],
        }
        self._write_json(source / "metadata" / "frame_0001.json", metadata)
        self._write_json(
            source / "occurrence_ground_truth" / "frame_0001.json",
            {"occurrences": [{"type": "animal_fallen", "animal_id": 17}]},
        )
        image = source / "generated_frames" / "frame_0001.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"synthetic png")
        return source

    def test_exports_sanitized_observable_and_private_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            destination = DatasetSnapshotExporter(
                source,
                "dataset_1",
                1,
                1,
                output_root=root / "output",
                allowed_source_root=root / "allowed",
            ).export()

            observable = json.loads(
                (destination / "observable/frames/frame_0001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(observable),
                {
                    "schema_version",
                    "frame",
                    "timestamp",
                    "day",
                    "hour",
                    "weather",
                    "detections",
                    "image_refs",
                },
            )
            self.assertEqual(
                set(observable["detections"][0]),
                {"track_id", "confidence", "bbox_xyxy", "center_xy"},
            )
            manifest_text = (destination / "observable_manifest.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("private", manifest_text)
            self.assertTrue(
                (destination / "private/ground_truth/frame_0001.json").is_file()
            )
            self.assertTrue((destination / "provenance/checksums.json").is_file())

    def test_rejects_source_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaises(ValueError):
                DatasetSnapshotExporter(
                    outside,
                    "dataset_1",
                    1,
                    1,
                    output_root=root / "output",
                    allowed_source_root=root / "allowed",
                )

    def test_never_overwrites_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            kwargs = {
                "output_root": root / "output",
                "allowed_source_root": root / "allowed",
            }
            DatasetSnapshotExporter(source, "dataset_1", 1, 1, **kwargs).export()
            with self.assertRaises(FileExistsError):
                DatasetSnapshotExporter(source, "dataset_1", 1, 1, **kwargs).export()

    def test_windows_permission_error_uses_verified_copy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            exporter = DatasetSnapshotExporter(
                source,
                "dataset_windows",
                1,
                1,
                output_root=root / "output",
                allowed_source_root=root / "allowed",
            )

            with patch(
                "src.analytical.dataset_exporter.os.replace",
                side_effect=PermissionError("diretório temporariamente bloqueado"),
            ), patch("src.analytical.dataset_exporter.time.sleep"):
                result = exporter.export()

            self.assertTrue((result / "observable_manifest.json").is_file())
            exporter._verify_checksums(result)

    def test_extracts_appearance_from_observable_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            image = Image.new("RGB", (60, 60), (70, 130, 65))
            draw = ImageDraw.Draw(image)
            draw.ellipse((20, 25, 40, 35), fill=(220, 210, 190))
            image.save(path)

            feature = DatasetSnapshotExporter._appearance_feature(
                path, [10, 10, 50, 50]
            )

        self.assertIsNotNone(feature)
        self.assertEqual(feature["source"], "raw_bbox_pixels")
        self.assertGreater(feature["contrast"], 20)


if __name__ == "__main__":
    unittest.main()
