import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DatasetSnapshotExporter:
    """Cria um snapshot v2, separando entradas observáveis e avaliação privada."""

    SCHEMA_VERSION = 2
    SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(
        self,
        source_root,
        dataset_id,
        start_frame,
        end_frame,
        *,
        include_raw=False,
        output_root=None,
        allowed_source_root=None,
        config_path=None,
    ):
        if not self.SAFE_ID.fullmatch(str(dataset_id)):
            raise ValueError("dataset_id deve conter apenas letras, números, _ ou -.")
        if int(start_frame) < 1 or int(end_frame) < int(start_frame):
            raise ValueError("Intervalo de frames inválido.")

        self.dataset_id = str(dataset_id)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.include_raw = bool(include_raw)
        self.source_root = self._resolve(source_root, PROJECT_ROOT)
        self.allowed_source_root = self._resolve(
            allowed_source_root or PROJECT_ROOT / "data", PROJECT_ROOT
        )
        self._require_within(self.source_root, self.allowed_source_root, "source_root")
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"Dataset de origem inexistente: {self.source_root}")

        self.output_root = self._resolve(
            output_root or PROJECT_ROOT / "data" / "datasets", PROJECT_ROOT
        )
        self.destination = self.output_root / self.dataset_id
        self.config_path = (
            self._resolve(config_path, PROJECT_ROOT) if config_path else None
        )

    @staticmethod
    def _resolve(path, relative_to):
        value = Path(path)
        if not value.is_absolute():
            value = relative_to / value
        return value.resolve()

    @staticmethod
    def _require_within(path, root, field):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field} deve permanecer dentro de {root}.") from exc

    def _source_file(self, directory, filename, *, required=True):
        directory_path = (self.source_root / directory).resolve()
        self._require_within(directory_path, self.source_root, directory)
        path = (directory_path / filename).resolve()
        self._require_within(path, directory_path, filename)
        if required and not path.is_file():
            raise FileNotFoundError(f"Arquivo obrigatório ausente: {path}")
        return path

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    @staticmethod
    def _sanitize_detection(detection):
        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Detecção sem bbox_xyxy válido.")
        return {
            "track_id": detection["id"],
            "confidence": detection["confidence"],
            "bbox_xyxy": bbox,
            "center_xy": [
                round((bbox[0] + bbox[2]) / 2, 2),
                round((bbox[1] + bbox[3]) / 2, 2),
            ],
        }

    @staticmethod
    def _appearance_feature(image_path, bbox):
        """Extrai cor do objeto a partir de pixels, sem usar estado interno."""

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                x1, y1, x2, y2 = (int(value) for value in bbox)
                x1 = max(0, min(image.width - 1, x1))
                y1 = max(0, min(image.height - 1, y1))
                x2 = max(x1 + 1, min(image.width, x2))
                y2 = max(y1 + 1, min(image.height, y2))
                crop = image.crop((x1, y1, x2, y2))
        except (OSError, ValueError):
            return None

        width, height = crop.size
        pixels = crop.load()
        border = []
        for x in range(width):
            border.append(pixels[x, 0])
            border.append(pixels[x, height - 1])
        for y in range(1, max(1, height - 1)):
            border.append(pixels[0, y])
            border.append(pixels[width - 1, y])
        if not border:
            return None
        background = tuple(
            sorted(pixel[channel] for pixel in border)[len(border) // 2]
            for channel in range(3)
        )

        cx1, cx2 = max(0, width // 5), min(width, width - width // 5)
        cy1, cy2 = max(0, height // 5), min(height, height - height // 5)
        central = [
            pixels[x, y]
            for y in range(cy1, cy2)
            for x in range(cx1, cx2)
        ]
        if not central:
            return None
        ranked = sorted(
            central,
            key=lambda pixel: sum(
                (pixel[channel] - background[channel]) ** 2
                for channel in range(3)
            ),
            reverse=True,
        )
        selected = ranked[: max(6, len(ranked) // 3)]
        foreground = tuple(
            round(sum(pixel[channel] for pixel in selected) / len(selected), 2)
            for channel in range(3)
        )
        contrast = math.sqrt(
            sum(
                (foreground[channel] - background[channel]) ** 2
                for channel in range(3)
            )
        )
        return {
            "foreground_rgb": list(foreground),
            "background_rgb": list(background),
            "contrast": round(contrast, 2),
            "source": "raw_bbox_pixels",
        }

    def _observable_frame(self, metadata, frame, raw_available, appearance_path):
        detections = []
        for item in metadata.get("detections", []):
            detection = self._sanitize_detection(item)
            appearance = self._appearance_feature(
                appearance_path,
                detection["bbox_xyxy"],
            )
            if appearance:
                detection["appearance"] = appearance
            detections.append(detection)
        image_refs = {
            "processed": f"images/processed/frame_{frame:04d}.png",
        }
        if raw_available:
            image_refs["raw"] = f"images/raw/frame_{frame:04d}.jpg"
        return {
            "schema_version": self.SCHEMA_VERSION,
            "frame": int(metadata["frame"]),
            "timestamp": metadata["timestamp"],
            "day": int(metadata["day"]),
            "hour": int(metadata["hora"]),
            "weather": {
                "temperature_c": metadata["temperatura"],
                "wind": metadata["vento"],
            },
            "detections": detections,
            "image_refs": image_refs,
        }

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_provenance_config(self, temporary):
        target = temporary / "provenance" / "config" / "config_snapshot.json"
        if self.config_path:
            if not self.config_path.is_file():
                raise FileNotFoundError(f"Configuração inexistente: {self.config_path}")
            payload = self._read_json(self.config_path)
        else:
            campaign_manifest = self.source_root / "campaign_manifest.json"
            if campaign_manifest.is_file():
                payload = self._read_json(campaign_manifest).get("config_snapshot", {})
            else:
                default_config = PROJECT_ROOT / "config" / "simulation_config.json"
                payload = self._read_json(default_config) if default_config.is_file() else {}
        self._write_json(target, payload)

    def _write_checksums(self, temporary):
        entries = {}
        checksum_path = temporary / "provenance" / "checksums.json"
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            if path == checksum_path:
                continue
            entries[path.relative_to(temporary).as_posix()] = self._sha256(path)
        self._write_json(
            checksum_path,
            {"algorithm": "sha256", "files": entries},
        )

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
                f"O snapshot {self.dataset_id} já existe e não será sobrescrito."
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        temporary = self.output_root / f".{self.dataset_id}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            for frame in range(self.start_frame, self.end_frame + 1):
                filename_json = f"frame_{frame:04d}.json"
                metadata_path = self._source_file("metadata", filename_json)
                processed_path = self._source_file(
                    "generated_frames", f"frame_{frame:04d}.png"
                )
                ground_truth_path = self._source_file(
                    "occurrence_ground_truth", filename_json
                )
                raw_path = self._source_file(
                    "raw_drone_frames", f"frame_{frame:04d}.jpg", required=False
                )
                if self.include_raw and not raw_path.is_file():
                    raise FileNotFoundError(f"Imagem raw solicitada e ausente: {raw_path}")

                metadata = self._read_json(metadata_path)
                if int(metadata.get("frame", -1)) != frame:
                    raise ValueError(f"Número divergente em {metadata_path}.")
                observable = self._observable_frame(
                    metadata,
                    frame,
                    self.include_raw,
                    raw_path if raw_path.is_file() else processed_path,
                )
                self._write_json(
                    temporary / "observable" / "frames" / filename_json,
                    observable,
                )
                processed_target = (
                    temporary
                    / "observable"
                    / "images"
                    / "processed"
                    / processed_path.name
                )
                processed_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(processed_path, processed_target)
                if self.include_raw:
                    raw_target = temporary / "observable" / "images" / "raw" / raw_path.name
                    raw_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(raw_path, raw_target)

                private_target = temporary / "private" / "ground_truth" / filename_json
                private_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ground_truth_path, private_target)

            observable_manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "dataset_id": self.dataset_id,
                "frame_range": {
                    "start": self.start_frame,
                    "end": self.end_frame,
                },
                "tracking_assumption": "synthetic_detector_track_id",
                "paths": {
                    "frames": "observable/frames",
                    "processed_images": "observable/images/processed",
                    **(
                        {"raw_images": "observable/images/raw"}
                        if self.include_raw
                        else {}
                    ),
                },
            }
            evaluation_manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "dataset_id": self.dataset_id,
                "observable_manifest": "observable_manifest.json",
                "ground_truth": "private/ground_truth",
                "checksums": "provenance/checksums.json",
            }
            self._write_json(temporary / "observable_manifest.json", observable_manifest)
            self._write_json(temporary / "evaluation_manifest.json", evaluation_manifest)
            self._write_provenance_config(temporary)
            self._write_checksums(temporary)
            self._publish(temporary)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.destination


def export_dataset_snapshot(*args, **kwargs):
    """Atalho funcional para integrações web e futuros comandos CLI."""

    return DatasetSnapshotExporter(*args, **kwargs).export()
