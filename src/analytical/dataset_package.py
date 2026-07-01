import json
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DatasetFrame:
    frame: int
    metadata_path: Path
    processed_image_path: Path
    raw_image_path: Path
    ground_truth_path: Path


class DatasetPackage:
    """Resolve e valida um dataset selecionado por manifesto."""

    SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = PROJECT_ROOT / self.manifest_path
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            self.manifest = json.load(file)
        self._validate_manifest()
        source_root = Path(self.manifest["source_root"])
        self.source_root = (
            source_root if source_root.is_absolute() else PROJECT_ROOT / source_root
        ).resolve()
        self.paths = self.manifest["paths"]
        self.dataset_id = self.manifest["dataset_id"]
        frame_range = self.manifest["frame_range"]
        self.start_frame = int(frame_range["start"])
        self.end_frame = int(frame_range["end"])

    def _validate_manifest(self):
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Versão de manifesto de dataset incompatível.")
        dataset_id = str(self.manifest.get("dataset_id", ""))
        if not self.SAFE_ID.fullmatch(dataset_id):
            raise ValueError("dataset_id deve conter apenas letras, números, _ ou -.")
        frame_range = self.manifest.get("frame_range", {})
        if int(frame_range.get("start", 0)) < 1:
            raise ValueError("frame_range.start deve ser maior ou igual a 1.")
        if int(frame_range.get("end", 0)) < int(frame_range.get("start", 0)):
            raise ValueError("frame_range.end deve ser maior ou igual a start.")
        required = {"metadata", "processed_images", "raw_images", "ground_truth"}
        missing = required - set(self.manifest.get("paths", {}))
        if missing:
            raise ValueError(f"Manifesto sem caminhos obrigatórios: {sorted(missing)}")

    def _path(self, key, filename):
        directory = (self.source_root / self.paths[key]).resolve()
        try:
            directory.relative_to(self.source_root)
        except ValueError as exc:
            raise ValueError(
                f"O caminho '{key}' deve permanecer dentro de source_root."
            ) from exc
        path = (directory / filename).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("Nome de arquivo fora do diretório do dataset.") from exc
        return path

    def frames(self):
        for frame in range(self.start_frame, self.end_frame + 1):
            item = DatasetFrame(
                frame=frame,
                metadata_path=self._path("metadata", f"frame_{frame:04d}.json"),
                processed_image_path=self._path(
                    "processed_images", f"frame_{frame:04d}.png"
                ),
                raw_image_path=self._path("raw_images", f"frame_{frame:04d}.jpg"),
                ground_truth_path=self._path(
                    "ground_truth", f"frame_{frame:04d}.json"
                ),
            )
            if not item.metadata_path.exists():
                raise FileNotFoundError(
                    f"Frame {frame} ausente no dataset {self.dataset_id}: "
                    f"{item.metadata_path}"
                )
            yield item

    @staticmethod
    def read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @classmethod
    def create_manifest(
        cls,
        dataset_id,
        source_root,
        start_frame,
        end_frame,
        output_path,
        description="",
    ):
        if not cls.SAFE_ID.fullmatch(dataset_id):
            raise ValueError("dataset_id inválido.")
        payload = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "description": description,
            "source_root": str(Path(source_root)),
            "frame_range": {"start": int(start_frame), "end": int(end_frame)},
            "paths": {
                "metadata": "metadata",
                "processed_images": "generated_frames",
                "raw_images": "raw_drone_frames",
                "ground_truth": "occurrence_ground_truth",
            },
        }
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path


@dataclass(frozen=True)
class ObservableFrame:
    frame: int
    observation_path: Path
    processed_image_path: Path
    raw_image_path: Path | None


class ObservableDatasetPackage:
    """Visao somente-leitura do contrato observavel v2.

    Este objeto nao conhece nem resolve caminhos de ground truth. Ele e a unica
    representacao de dataset que deve ser entregue ao agente analitico.
    """

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = PROJECT_ROOT / self.manifest_path
        self.manifest_path = self.manifest_path.resolve()
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            self.manifest = json.load(file)
        self.root = self.manifest_path.parent
        self._validate_manifest()
        self.dataset_id = self.manifest["dataset_id"]
        self.paths = self.manifest["paths"]
        frame_range = self.manifest["frame_range"]
        self.start_frame = int(frame_range["start"])
        self.end_frame = int(frame_range["end"])

    def _validate_manifest(self):
        if self.manifest.get("schema_version") != 2:
            raise ValueError("Versao de manifesto observavel incompatível.")
        dataset_id = str(self.manifest.get("dataset_id", ""))
        if not DatasetPackage.SAFE_ID.fullmatch(dataset_id):
            raise ValueError("dataset_id invalido.")
        serialized = json.dumps(self.manifest).lower()
        if "private" in serialized or "ground_truth" in serialized:
            raise ValueError("Manifesto observavel nao pode referenciar dados privados.")
        frame_range = self.manifest.get("frame_range", {})
        start = int(frame_range.get("start", 0))
        end = int(frame_range.get("end", 0))
        if start < 1 or end < start:
            raise ValueError("Intervalo de frames invalido.")
        required = {"frames", "processed_images"}
        missing = required - set(self.manifest.get("paths", {}))
        if missing:
            raise ValueError(f"Manifesto observavel incompleto: {sorted(missing)}")

    def _path(self, key, filename):
        directory = (self.root / self.paths[key]).resolve()
        try:
            directory.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Caminho observavel invalido: {key}") from exc
        path = (directory / filename).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("Arquivo fora do dataset observavel.") from exc
        return path

    def frames(self):
        for frame in range(self.start_frame, self.end_frame + 1):
            observation_path = self._path("frames", f"frame_{frame:04d}.json")
            processed_path = self._path(
                "processed_images", f"frame_{frame:04d}.png"
            )
            raw_path = (
                self._path("raw_images", f"frame_{frame:04d}.jpg")
                if "raw_images" in self.paths
                else None
            )
            if not observation_path.is_file() or not processed_path.is_file():
                raise FileNotFoundError(f"Frame observavel {frame} incompleto.")
            yield ObservableFrame(
                frame=frame,
                observation_path=observation_path,
                processed_image_path=processed_path,
                raw_image_path=raw_path,
            )

    @staticmethod
    def read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


@dataclass(frozen=True)
class EvaluationFrame:
    frame: int
    metadata_path: Path
    ground_truth_path: Path


class EvaluationDatasetPackage:
    """Visao privada usada exclusivamente pela etapa de avaliacao."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = PROJECT_ROOT / self.manifest_path
        self.manifest_path = self.manifest_path.resolve()
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            self.manifest = json.load(file)
        if self.manifest.get("schema_version") != 2:
            raise ValueError("Versao de manifesto de avaliacao incompatível.")
        self.root = self.manifest_path.parent
        observable_path = (self.root / self.manifest["observable_manifest"]).resolve()
        self.observable = ObservableDatasetPackage(observable_path)
        self.dataset_id = self.observable.dataset_id
        self.start_frame = self.observable.start_frame
        self.end_frame = self.observable.end_frame
        self.ground_truth_dir = (self.root / self.manifest["ground_truth"]).resolve()
        try:
            self.ground_truth_dir.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Ground truth fora do snapshot.") from exc

    def frames(self):
        for item in self.observable.frames():
            ground_truth_path = (
                self.ground_truth_dir / f"frame_{item.frame:04d}.json"
            ).resolve()
            try:
                ground_truth_path.relative_to(self.ground_truth_dir)
            except ValueError as exc:
                raise ValueError("Ground truth fora do diretorio privado.") from exc
            if not ground_truth_path.is_file():
                raise FileNotFoundError(
                    f"Ground truth ausente para o frame {item.frame}."
                )
            yield EvaluationFrame(
                frame=item.frame,
                metadata_path=item.observation_path,
                ground_truth_path=ground_truth_path,
            )

    @staticmethod
    def read_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
