import hashlib
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from src.analytical.dataset_exporter import PROJECT_ROOT


class CompositeDatasetExporter:
    """Compõe snapshots v2 contíguos sem alterar os artefatos de origem."""

    SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(
        self,
        base_snapshot,
        extension_snapshot,
        dataset_id,
        *,
        output_root=None,
    ):
        if not self.SAFE_ID.fullmatch(str(dataset_id)):
            raise ValueError("dataset_id composto inválido.")
        self.base_root = Path(base_snapshot).resolve()
        self.extension_root = Path(extension_snapshot).resolve()
        self.dataset_id = str(dataset_id)
        self.output_root = Path(
            output_root or PROJECT_ROOT / "data" / "datasets"
        ).resolve()
        self.destination = self.output_root / self.dataset_id
        self.base_manifest = self._read_json(
            self.base_root / "observable_manifest.json"
        )
        self.extension_manifest = self._read_json(
            self.extension_root / "observable_manifest.json"
        )
        self._validate()

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def _validate(self):
        for root, manifest in (
            (self.base_root, self.base_manifest),
            (self.extension_root, self.extension_manifest),
        ):
            if manifest.get("schema_version") != 2:
                raise ValueError(f"Snapshot incompatível: {root}")
            if not (root / "evaluation_manifest.json").is_file():
                raise FileNotFoundError(f"Manifesto privado ausente: {root}")
        base_end = int(self.base_manifest["frame_range"]["end"])
        extension_start = int(self.extension_manifest["frame_range"]["start"])
        if extension_start != base_end + 1:
            raise ValueError(
                "Snapshots não são contíguos: a extensão deve iniciar no frame "
                f"{base_end + 1}."
            )

    @staticmethod
    def _copy_directory(source, destination):
        if not source.is_dir():
            raise FileNotFoundError(f"Diretório ausente: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if not path.is_file():
                continue
            target = destination / path.name
            if target.exists():
                raise ValueError(f"Colisão de artefato composto: {target.name}")
            shutil.copy2(path, target)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_checksums(self, root):
        checksum_path = root / "provenance" / "checksums.json"
        files = {
            path.relative_to(root).as_posix(): self._sha256(path)
            for path in sorted(item for item in root.rglob("*") if item.is_file())
            if path != checksum_path
        }
        self._write_json(checksum_path, {"algorithm": "sha256", "files": files})

    def _verify_checksums(self, root):
        payload = self._read_json(root / "provenance" / "checksums.json")
        for relative_path, expected in payload["files"].items():
            path = root / relative_path
            if not path.is_file() or self._sha256(path) != expected:
                raise IOError(f"Falha ao verificar artefato publicado: {relative_path}")

    def _publish(self, temporary):
        """Publica com tolerância a bloqueios transitórios de diretório no Windows."""
        last_error = None
        for attempt in range(6):
            try:
                os.replace(temporary, self.destination)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.15 * (attempt + 1))

        try:
            shutil.copytree(temporary, self.destination)
            self._verify_checksums(self.destination)
            shutil.rmtree(temporary, ignore_errors=True)
        except Exception:
            shutil.rmtree(self.destination, ignore_errors=True)
            raise last_error

    def export(self):
        if self.destination.exists():
            raise FileExistsError(
                f"O dataset composto {self.dataset_id} já existe."
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        temporary = self.output_root / f".{self.dataset_id}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            for root in (self.base_root, self.extension_root):
                self._copy_directory(
                    root / "observable" / "frames",
                    temporary / "observable" / "frames",
                )
                self._copy_directory(
                    root / "observable" / "images" / "processed",
                    temporary / "observable" / "images" / "processed",
                )
                self._copy_directory(
                    root / "private" / "ground_truth",
                    temporary / "private" / "ground_truth",
                )

            include_raw = all(
                "raw_images" in manifest["paths"]
                for manifest in (self.base_manifest, self.extension_manifest)
            )
            if include_raw:
                for root in (self.base_root, self.extension_root):
                    self._copy_directory(
                        root / "observable" / "images" / "raw",
                        temporary / "observable" / "images" / "raw",
                    )

            frame_range = {
                "start": int(self.base_manifest["frame_range"]["start"]),
                "end": int(self.extension_manifest["frame_range"]["end"]),
            }
            observable_manifest = {
                "schema_version": 2,
                "dataset_id": self.dataset_id,
                "frame_range": frame_range,
                "tracking_assumption": self.extension_manifest.get(
                    "tracking_assumption", "synthetic_detector_track_id"
                ),
                "paths": {
                    "frames": "observable/frames",
                    "processed_images": "observable/images/processed",
                    **(
                        {"raw_images": "observable/images/raw"}
                        if include_raw
                        else {}
                    ),
                },
            }
            evaluation_manifest = {
                "schema_version": 2,
                "dataset_id": self.dataset_id,
                "observable_manifest": "observable_manifest.json",
                "ground_truth": "private/ground_truth",
                "checksums": "provenance/checksums.json",
            }
            lineage = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "composition": "parent_plus_extension",
                "parent_dataset_id": self.base_manifest["dataset_id"],
                "parent_frame_range": self.base_manifest["frame_range"],
                "extension_dataset_id": self.extension_manifest["dataset_id"],
                "extension_frame_range": self.extension_manifest["frame_range"],
                "result_dataset_id": self.dataset_id,
                "result_frame_range": frame_range,
            }
            self._write_json(
                temporary / "observable_manifest.json", observable_manifest
            )
            self._write_json(
                temporary / "evaluation_manifest.json", evaluation_manifest
            )
            self._write_json(temporary / "provenance" / "lineage.json", lineage)
            extension_config = (
                self.extension_root
                / "provenance"
                / "config"
                / "config_snapshot.json"
            )
            if extension_config.is_file():
                target = (
                    temporary
                    / "provenance"
                    / "config"
                    / "config_snapshot.json"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extension_config, target)
            self._write_checksums(temporary)
            self._publish(temporary)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.destination
