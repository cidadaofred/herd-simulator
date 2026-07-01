import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analytical.composite_dataset import CompositeDatasetExporter


class CompositeDatasetExporterTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _snapshot(self, root, dataset_id, frame_start, frame_end):
        snapshot = root / dataset_id
        paths = {
            "frames": "observable/frames",
            "processed_images": "observable/images/processed",
        }
        self._write_json(
            snapshot / "observable_manifest.json",
            {
                "schema_version": 2,
                "dataset_id": dataset_id,
                "frame_range": {"start": frame_start, "end": frame_end},
                "tracking_assumption": "synthetic_detector_track_id",
                "paths": paths,
            },
        )
        self._write_json(
            snapshot / "evaluation_manifest.json",
            {
                "schema_version": 2,
                "dataset_id": dataset_id,
                "observable_manifest": "observable_manifest.json",
                "ground_truth": "private/ground_truth",
            },
        )
        self._write_json(
            snapshot / "provenance/config/config_snapshot.json",
            {"campaign": {"campaign_id": dataset_id}},
        )
        for frame in range(frame_start, frame_end + 1):
            name = f"frame_{frame:04d}"
            self._write_json(
                snapshot / "observable/frames" / f"{name}.json",
                {"frame": frame},
            )
            self._write_json(
                snapshot / "private/ground_truth" / f"{name}.json",
                {"frame": frame, "occurrences": []},
            )
            image = snapshot / "observable/images/processed" / f"{name}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{frame}".encode())
        return snapshot

    def test_composes_contiguous_snapshots_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._snapshot(root / "sources", "base", 1, 2)
            extension = self._snapshot(root / "sources", "extension", 3, 4)
            original = (base / "observable/frames/frame_0001.json").read_bytes()

            result = CompositeDatasetExporter(
                base,
                extension,
                "composed",
                output_root=root / "datasets",
            ).export()

            manifest = json.loads(
                (result / "observable_manifest.json").read_text(encoding="utf-8")
            )
            lineage = json.loads(
                (result / "provenance/lineage.json").read_text(encoding="utf-8")
            )
            frames = sorted((result / "observable/frames").glob("frame_*.json"))
            self.assertEqual(manifest["frame_range"], {"start": 1, "end": 4})
            self.assertEqual(len(frames), 4)
            self.assertEqual(lineage["parent_dataset_id"], "base")
            self.assertEqual(lineage["extension_dataset_id"], "extension")
            self.assertEqual(
                (base / "observable/frames/frame_0001.json").read_bytes(),
                original,
            )
            self.assertTrue((result / "provenance/checksums.json").is_file())

    def test_rejects_non_contiguous_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._snapshot(root, "base", 1, 2)
            extension = self._snapshot(root, "extension", 4, 5)

            with self.assertRaisesRegex(ValueError, "contíguos"):
                CompositeDatasetExporter(
                    base,
                    extension,
                    "invalid",
                    output_root=root / "datasets",
                )

    def test_never_overwrites_composed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._snapshot(root / "sources", "base", 1, 1)
            extension = self._snapshot(root / "sources", "extension", 2, 2)
            kwargs = {"output_root": root / "datasets"}
            CompositeDatasetExporter(base, extension, "result", **kwargs).export()

            with self.assertRaises(FileExistsError):
                CompositeDatasetExporter(base, extension, "result", **kwargs).export()

    def test_windows_permission_error_uses_verified_copy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._snapshot(root / "sources", "base", 1, 1)
            extension = self._snapshot(root / "sources", "extension", 2, 2)
            exporter = CompositeDatasetExporter(
                base,
                extension,
                "result",
                output_root=root / "datasets",
            )

            with patch(
                "src.analytical.composite_dataset.os.replace",
                side_effect=PermissionError("diretório temporariamente bloqueado"),
            ), patch("src.analytical.composite_dataset.time.sleep"):
                result = exporter.export()

            self.assertTrue((result / "observable_manifest.json").is_file())
            self.assertTrue((result / "provenance/checksums.json").is_file())


if __name__ == "__main__":
    unittest.main()
