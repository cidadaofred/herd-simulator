import copy
import json
import random
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from src.analytical.analytical_agent import AnalyticalAgent
from src.analytical.composite_dataset import CompositeDatasetExporter
from src.analytical.dataset_exporter import DatasetSnapshotExporter
from src.analytical.dataset_package import (
    DatasetPackage,
    EvaluationDatasetPackage,
    ObservableDatasetPackage,
)
from src.analytical.evaluator import Evaluator
from src.coordinator.coordinator_agent import CoordinatorAgent
from src.main import PROJECT_ROOT, load_config
from src.simulator.simulator_agent import SimulatorAgent


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class SpecialistSimulationRequest:
    campaign_id: str
    num_days: int = 1
    num_cattle: int = 40
    seed: int = 42
    start_date: str = "2026-06-20"
    temperature_offset_c: int = 0
    event_types: tuple[str, ...] = ()
    user_id: str = "web"


@dataclass(frozen=True)
class SimulationRunResult:
    campaign_id: str
    source_root: Path
    frame_start: int
    frame_end: int
    frame_count: int
    snapshot_id: str
    snapshot_root: Path
    observable_manifest: Path
    evaluation_manifest: Path
    preview_image: Path


@dataclass(frozen=True)
class AnalysisRunResult:
    dataset_id: str
    run_id: str
    output_dir: Path
    summary_path: Path
    alerts_path: Path
    evaluation_path: Path


@dataclass(frozen=True)
class ContinuationRequest:
    base_dataset_id: str
    new_dataset_id: str
    additional_days: int
    target_active_animals: int
    seed: int = 42
    event_types: tuple[str, ...] = ()
    random_event_count: int = 0


@dataclass(frozen=True)
class ContinuationRunResult:
    parent_dataset_id: str
    dataset_id: str
    branch_campaign_id: str
    parent_frame_end: int
    extension_frame_start: int
    extension_frame_end: int
    total_frame_count: int
    active_animals_before: int
    active_animals_requested: int
    snapshot_root: Path
    segment_root: Path
    observable_manifest: Path
    evaluation_manifest: Path
    preview_image: Path


@dataclass(frozen=True)
class DatasetDeletionPreview:
    requested_dataset_id: str
    dataset_ids: tuple[str, ...]
    analysis_run_count: int
    segment_count: int
    campaign_count: int
    export_count: int
    total_bytes: int


@dataclass(frozen=True)
class DatasetDeletionResult:
    deleted_dataset_ids: tuple[str, ...]
    deleted_path_count: int
    reclaimed_bytes: int


@dataclass(frozen=True)
class AnalysisDeletionPreview:
    dataset_id: str
    run_id: str
    output_bytes: int
    export_count: int
    total_bytes: int


@dataclass(frozen=True)
class AnalysisDeletionResult:
    dataset_id: str
    run_id: str
    deleted_path_count: int
    reclaimed_bytes: int


@dataclass(frozen=True)
class CoordinationRunResult:
    dataset_id: str
    run_id: str
    analysis_dir: Path
    report_path: Path
    decision: str
    decision_label: str


class ExperimentService:
    """Fachada segura para o ciclo simular, empacotar, inferir e avaliar."""

    EVENT_LABELS = {
        "extreme_heat": "Calor extremo",
        "animal_fallen": "Animal caído",
        "extreme_cold": "Frio extremo",
        "animal_missing": "Animal desaparecido",
        "animal_parturition": "Vaca em trabalho de parto",
        "animal_death": "Morte de animal",
    }

    ANALYSIS_MODELS = {
        "modelo_regras_estatistico": {
            "label": "Modelo de regras estatistico",
            "description": (
                "Motor atual: regras temporais, limiares estatisticos e "
                "consistencia de tracks para gerar alertas."
            ),
        }
    }

    def __init__(self, project_root=PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.data_root = self.project_root / "data"

    @staticmethod
    def _validate_id(value, field):
        value = str(value).strip()
        if not SAFE_ID.fullmatch(value):
            raise ValueError(
                f"{field} deve conter apenas letras, números, '_' ou '-'."
            )
        return value

    def analysis_model_options(self):
        return [
            {"id": model_id, **values}
            for model_id, values in self.ANALYSIS_MODELS.items()
        ]

    def _analysis_run_id(self, model_id):
        model_id = self._validate_id(model_id, "analysis_model_id")
        if model_id not in self.ANALYSIS_MODELS:
            raise ValueError("Modelo de analise nao reconhecido.")
        return self._validate_id(
            f"web_{model_id}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
            "run_id",
        )

    @staticmethod
    def _validate_request(request):
        if not 1 <= int(request.num_days) <= 30:
            raise ValueError("num_days deve estar entre 1 e 30.")
        if not 4 <= int(request.num_cattle) <= 100:
            raise ValueError("num_cattle deve estar entre 4 e 100.")
        if not -15 <= int(request.temperature_offset_c) <= 15:
            raise ValueError("temperature_offset_c deve estar entre -15 e 15.")
        date.fromisoformat(str(request.start_date))

    @staticmethod
    def _allocate_category_counts(categories, total):
        names = list(categories)
        original_total = sum(int(categories[name]["count"]) for name in names)
        raw = {
            name: total * int(categories[name]["count"]) / original_total
            for name in names
        }
        counts = {name: max(1, int(raw[name])) for name in names}
        while sum(counts.values()) > total:
            candidates = [name for name in names if counts[name] > 1]
            name = min(candidates, key=lambda item: raw[item] - counts[item])
            counts[name] -= 1
        while sum(counts.values()) < total:
            name = max(names, key=lambda item: raw[item] - counts[item])
            counts[name] += 1
        return counts

    def build_config(self, request: SpecialistSimulationRequest):
        self._validate_request(request)
        campaign_id = self._validate_id(request.campaign_id, "campaign_id")
        user_id = self._validate_id(request.user_id, "user_id")
        config = copy.deepcopy(load_config())

        config["simulation"].update(
            {
                "num_days": int(request.num_days),
                "num_bovinos": int(request.num_cattle),
                "seed": int(request.seed),
            }
        )
        config["campaign"].update(
            {
                "enabled": True,
                "user_id": user_id,
                "campaign_id": campaign_id,
                "mode": "new",
                "start_date": str(request.start_date),
                "allow_overwrite": False,
            }
        )

        categories = config["nutrition"]["categories"]
        counts = self._allocate_category_counts(categories, int(request.num_cattle))
        for name, count in counts.items():
            categories[name]["count"] = count
            categories[name]["cohort_count"] = min(
                max(1, int(categories[name]["cohort_count"])), count
            )

        category_ids = {}
        next_id = 1
        for name, values in categories.items():
            count = int(values["count"])
            category_ids[name] = list(range(next_id, next_id + count))
            next_id += count

        base_lot = copy.deepcopy(config["social_behavior"]["lots"][0])
        base_lot.update(
            {
                "id": "lote_base",
                "animal_ids": f"1-{int(request.num_cattle)}",
                "arrival_day": 1,
                "enabled": True,
            }
        )
        config["social_behavior"]["lots"] = [base_lot]

        offset = int(request.temperature_offset_c)
        temperature_by_hour = config["weather"]["temperature_by_hour"]
        for hour, value in list(temperature_by_hour.items()):
            if not str(hour).startswith("_"):
                temperature_by_hour[hour] = float(value) + offset

        requested_events = set(request.event_types)
        known_events = set(self.EVENT_LABELS)
        if not requested_events <= known_events:
            raise ValueError("Tipo de evento não reconhecido.")
        schedule = [
            copy.deepcopy(event)
            for event in config["events"].get("schedule", [])
            if event.get("type") in requested_events
            and int(event.get("day", 1)) <= int(request.num_days)
        ]
        lactating = category_ids.get("vaca_lactante", [])
        adults = category_ids.get("vaca_adulta", []) or category_ids.get("novilha", [])
        all_ids = list(range(1, int(request.num_cattle) + 1))
        for event in schedule:
            if event["type"] == "animal_parturition":
                event["animal_id"] = lactating[0] if lactating else all_ids[0]
                event.setdefault("parameters", {})["calf_id"] = int(
                    request.num_cattle
                ) + 1
            elif event["type"] == "animal_death":
                event["animal_id"] = adults[0] if adults else all_ids[-1]
            elif event["type"] == "animal_fallen":
                event["animal_id"] = adults[-1] if adults else all_ids[-1]
            elif event["type"] == "animal_missing":
                event["animal_id"] = all_ids[-1]
        config["events"]["enabled"] = bool(schedule)
        config["events"]["schedule"] = schedule
        return config

    def config_preview(self, request):
        config = self.build_config(request)
        return {
            "campaign": config["campaign"],
            "simulation": config["simulation"],
            "herd_categories": {
                name: values["count"]
                for name, values in config["nutrition"]["categories"].items()
            },
            "events": config["events"],
            "temperature_by_hour": config["weather"]["temperature_by_hour"],
        }

    def run_simulation(self, request: SpecialistSimulationRequest):
        config = self.build_config(request)
        simulator = SimulatorAgent(config)
        generated = simulator.run()
        metadata_files = sorted(simulator.metadata_dir.glob("frame_*.json"))
        if not generated or not metadata_files:
            raise RuntimeError("O simulador não produziu frames.")
        frame_numbers = [int(path.stem.split("_")[-1]) for path in metadata_files]
        snapshot_id = self._validate_id(request.campaign_id, "snapshot_id")
        snapshot_root = DatasetSnapshotExporter(
            simulator.campaign_store.root,
            snapshot_id,
            min(frame_numbers),
            max(frame_numbers),
            allowed_source_root=self.data_root,
        ).export()
        return SimulationRunResult(
            campaign_id=simulator.campaign_store.campaign_id,
            source_root=simulator.campaign_store.root,
            frame_start=min(frame_numbers),
            frame_end=max(frame_numbers),
            frame_count=len(frame_numbers),
            snapshot_id=snapshot_id,
            snapshot_root=snapshot_root,
            observable_manifest=snapshot_root / "observable_manifest.json",
            evaluation_manifest=snapshot_root / "evaluation_manifest.json",
            preview_image=Path(generated[-1]),
        )

    def list_snapshots(self):
        datasets_root = self.data_root / "datasets"
        if not datasets_root.exists():
            return []
        return [
            path.parent.name
            for path in sorted(datasets_root.glob("*/observable_manifest.json"))
        ]

    def snapshot_paths(self, dataset_id):
        dataset_id = self._validate_id(dataset_id, "dataset_id")
        root = (self.data_root / "datasets" / dataset_id).resolve()
        try:
            root.relative_to((self.data_root / "datasets").resolve())
        except ValueError as exc:
            raise ValueError("Dataset fora do catálogo.") from exc
        observable = root / "observable_manifest.json"
        evaluation = root / "evaluation_manifest.json"
        if not observable.is_file() or not evaluation.is_file():
            raise FileNotFoundError(f"Snapshot incompleto: {dataset_id}")
        return observable, evaluation

    def _snapshot_context(self, dataset_id):
        observable, evaluation = self.snapshot_paths(dataset_id)
        root = observable.parent
        config_path = root / "provenance" / "config" / "config_snapshot.json"
        if not config_path.is_file():
            raise ValueError("O dataset não possui configuração de proveniência.")
        config = DatasetPackage.read_json(config_path)
        campaign = config.get("campaign", {})
        if not campaign.get("enabled"):
            raise ValueError("O dataset não pertence a uma campanha continuável.")
        user_id = self._validate_id(campaign.get("user_id"), "user_id")
        campaign_id = self._validate_id(campaign.get("campaign_id"), "campaign_id")
        campaign_root = (
            self.data_root / "users" / user_id / "campaigns" / campaign_id
        ).resolve()
        checkpoint_path = campaign_root / "checkpoints" / "latest.json"
        if not checkpoint_path.is_file():
            raise ValueError("A campanha de origem não possui checkpoint final.")
        checkpoint = DatasetPackage.read_json(checkpoint_path)
        manifest = DatasetPackage.read_json(observable)
        if int(manifest["frame_range"]["end"]) != int(checkpoint["global_frame"]):
            raise ValueError(
                "O dataset não termina no checkpoint mais recente da campanha."
            )
        return {
            "observable": observable,
            "evaluation": evaluation,
            "snapshot_root": root,
            "config": config,
            "campaign_root": campaign_root,
            "campaign_id": campaign_id,
            "user_id": user_id,
            "checkpoint_path": checkpoint_path,
            "checkpoint": checkpoint,
            "manifest": manifest,
        }

    def list_continuable_snapshots(self):
        result = []
        for dataset_id in self.list_snapshots():
            try:
                context = self._snapshot_context(dataset_id)
            except (ValueError, FileNotFoundError, KeyError):
                continue
            inventory = context["checkpoint"].get("inventory_status", {})
            result.append(
                {
                    "dataset_id": dataset_id,
                    "completed_day": int(context["checkpoint"]["completed_day"]),
                    "frame_end": int(context["checkpoint"]["global_frame"]),
                    "active_animals": sum(
                        value == "active" for value in inventory.values()
                    ),
                    "campaign_id": context["campaign_id"],
                }
            )
        return result

    @staticmethod
    def _event_animal_id(event_type, active_ids, catalog, used_ids):
        available = [item for item in active_ids if item not in used_ids]
        if not available:
            available = list(active_ids)
        if not available:
            return None
        if event_type == "animal_parturition":
            lactating = [
                item
                for item in available
                if catalog.get(str(item), {}).get("category") == "vaca_lactante"
            ]
            return (lactating or available)[0]
        return available[-1] if event_type == "animal_missing" else available[0]

    def _random_future_events(
        self,
        config,
        checkpoint,
        request,
        active_ids,
        first_new_id,
    ):
        if not request.event_types or request.random_event_count <= 0:
            return []
        templates = {
            item["type"]: item
            for item in load_config()["events"].get("schedule", [])
            if item["type"] in self.EVENT_LABELS
        }
        selected_types = [
            item for item in request.event_types if item in templates
        ]
        if not selected_types:
            return []
        rng = random.Random(int(request.seed) + int(checkpoint["global_frame"]))
        first_day = int(checkpoint["completed_day"]) + 1
        last_day = first_day + int(request.additional_days) - 1
        hours = [int(value) for value in config["simulation"]["frame_hours"]]
        catalog = checkpoint.get("animal_catalog", {})
        events = []
        used_ids = set()
        calf_id = int(first_new_id)
        for index in range(int(request.random_event_count)):
            event_type = rng.choice(selected_types)
            event = copy.deepcopy(templates[event_type])
            event["id"] = f"BRANCH_{event_type.upper()}_{index + 1:02d}"
            event["day"] = rng.randint(first_day, last_day)
            duration = max(1, int(event.get("duration_hours", 1)))
            required_observations = min(duration, len(hours))
            eligible_hours = [
                start_hour
                for start_hour in hours
                if sum(
                    start_hour <= observed_hour < start_hour + duration
                    for observed_hour in hours
                )
                >= required_observations
            ]
            event["hour"] = rng.choice(eligible_hours or hours)
            animal_id = self._event_animal_id(
                event_type, active_ids, catalog, used_ids
            )
            if "animal_id" in event or event_type.startswith("animal_"):
                if animal_id is None:
                    continue
                event["animal_id"] = animal_id
                used_ids.add(animal_id)
            if event_type == "animal_parturition":
                event.setdefault("parameters", {})["calf_id"] = calf_id
                calf_id += 1
            events.append(event)
        return sorted(events, key=lambda item: (item["day"], item["hour"]))

    def continue_experiment(self, request: ContinuationRequest):
        if not 1 <= int(request.additional_days) <= 30:
            raise ValueError("additional_days deve estar entre 1 e 30.")
        if not 1 <= int(request.target_active_animals) <= 200:
            raise ValueError("target_active_animals deve estar entre 1 e 200.")
        parent_id = self._validate_id(request.base_dataset_id, "base_dataset_id")
        dataset_id = self._validate_id(request.new_dataset_id, "new_dataset_id")
        if parent_id == dataset_id:
            raise ValueError("A continuação deve possuir um novo dataset_id.")
        context = self._snapshot_context(parent_id)
        checkpoint = context["checkpoint"]
        config = copy.deepcopy(context["config"])
        completed_day = int(checkpoint["completed_day"])
        parent_frame_end = int(checkpoint["global_frame"])
        first_day = completed_day + 1
        first_hour = int(config["simulation"]["frame_hours"][0])
        branch_campaign_id = self._validate_id(
            f"branch_{dataset_id}", "branch_campaign_id"
        )
        config["campaign"].update(
            {
                "enabled": True,
                "user_id": context["user_id"],
                "campaign_id": branch_campaign_id,
                "mode": "branch",
                "source_campaign_id": context["campaign_id"],
                "source_checkpoint": context["checkpoint_path"].name,
                "allow_overwrite": False,
            }
        )
        config["simulation"]["num_days"] = int(request.additional_days)
        config["simulation"]["seed"] = int(request.seed)

        inventory = {
            int(key): value
            for key, value in checkpoint.get("inventory_status", {}).items()
        }
        active_ids = sorted(
            cattle_id for cattle_id, status in inventory.items() if status == "active"
        )
        active_before = len(active_ids)
        known_ids = {
            int(key) for key in checkpoint.get("animal_catalog", {})
        } | set(inventory)
        config["campaign"]["checkpoint_animal_ids"] = sorted(known_ids)
        next_id = max(known_ids, default=0) + 1
        administrative_schedule = []
        if request.target_active_animals > active_before:
            count = int(request.target_active_animals) - active_before
            preferred_shelter = (
                "P4"
                if "P4" in self.farm_paddock_ids(config)
                else config["behavior_planner"]["initial_home_paddock"]
            )
            animals = [
                {
                    "id": next_id + index,
                    "category": "novilha",
                    "cohort": f"novilha_branch_{dataset_id}",
                    "lot_id": f"lote_branch_{dataset_id}",
                    "preferred_shelter": preferred_shelter,
                }
                for index in range(count)
            ]
            administrative_schedule.append(
                {
                    "id": f"ENTRADA_{dataset_id}",
                    "type": "animal_entry",
                    "day": first_day,
                    "hour": first_hour,
                    "animals": animals,
                }
            )
            active_ids.extend(item["id"] for item in animals)
            next_id += count
        elif request.target_active_animals < active_before:
            count = active_before - int(request.target_active_animals)
            sold_ids = sorted(active_ids, reverse=True)[:count]
            administrative_schedule.append(
                {
                    "id": f"VENDA_{dataset_id}",
                    "type": "animal_sale",
                    "day": first_day,
                    "hour": first_hour,
                    "animal_ids": sold_ids,
                }
            )
            active_ids = [item for item in active_ids if item not in sold_ids]

        config["administrative_events"]["enabled"] = bool(
            administrative_schedule
        )
        config["administrative_events"]["schedule"] = administrative_schedule
        future_events = self._random_future_events(
            config,
            checkpoint,
            request,
            active_ids,
            next_id,
        )
        config["events"]["enabled"] = bool(future_events)
        config["events"]["schedule"] = future_events

        simulator = SimulatorAgent(config)
        generated = simulator.run()
        if not generated:
            raise RuntimeError("A continuação não gerou frames.")
        extension_start = parent_frame_end + 1
        extension_end = simulator.global_frame
        segments_root = self.data_root / "segments"
        segment_id = f"segment_{dataset_id}"
        segment_root = DatasetSnapshotExporter(
            simulator.campaign_store.root,
            segment_id,
            extension_start,
            extension_end,
            output_root=segments_root,
            allowed_source_root=self.data_root,
        ).export()
        snapshot_root = CompositeDatasetExporter(
            context["snapshot_root"],
            segment_root,
            dataset_id,
        ).export()
        return ContinuationRunResult(
            parent_dataset_id=parent_id,
            dataset_id=dataset_id,
            branch_campaign_id=branch_campaign_id,
            parent_frame_end=parent_frame_end,
            extension_frame_start=extension_start,
            extension_frame_end=extension_end,
            total_frame_count=extension_end
            - int(context["manifest"]["frame_range"]["start"])
            + 1,
            active_animals_before=active_before,
            active_animals_requested=int(request.target_active_animals),
            snapshot_root=snapshot_root,
            segment_root=segment_root,
            observable_manifest=snapshot_root / "observable_manifest.json",
            evaluation_manifest=snapshot_root / "evaluation_manifest.json",
            preview_image=Path(generated[-1]),
        )

    @staticmethod
    def farm_paddock_ids(config):
        return {item["id"] for item in config["environment"]["paddocks"]}

    @staticmethod
    def _path_size(path):
        path = Path(path)
        if path.is_file():
            return path.stat().st_size
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )

    def _deletion_scope(self, dataset_id):
        dataset_id = self._validate_id(dataset_id, "dataset_id")
        datasets_root = (self.data_root / "datasets").resolve()
        requested_root = datasets_root / dataset_id
        if not (requested_root / "observable_manifest.json").is_file():
            raise FileNotFoundError(f"Dataset inexistente: {dataset_id}")

        dataset_roots = {
            path.parent.name: path.parent.resolve()
            for path in datasets_root.glob("*/observable_manifest.json")
        }
        parent_by_dataset = {}
        for current_id, root in dataset_roots.items():
            lineage_path = root / "provenance" / "lineage.json"
            if lineage_path.is_file():
                lineage = DatasetPackage.read_json(lineage_path)
                parent_by_dataset[current_id] = lineage.get("parent_dataset_id")

        target_ids = {dataset_id}
        changed = True
        while changed:
            changed = False
            for current_id, parent_id in parent_by_dataset.items():
                if parent_id in target_ids and current_id not in target_ids:
                    target_ids.add(current_id)
                    changed = True

        campaign_by_dataset = {}
        segment_paths = set()
        for current_id, root in dataset_roots.items():
            config_path = root / "provenance" / "config" / "config_snapshot.json"
            if config_path.is_file():
                config = DatasetPackage.read_json(config_path)
                campaign = config.get("campaign", {})
                user_id = campaign.get("user_id")
                campaign_id = campaign.get("campaign_id")
                if user_id and campaign_id:
                    campaign_by_dataset[current_id] = (
                        self._validate_id(user_id, "user_id"),
                        self._validate_id(campaign_id, "campaign_id"),
                    )
            lineage_path = root / "provenance" / "lineage.json"
            if current_id in target_ids and lineage_path.is_file():
                lineage = DatasetPackage.read_json(lineage_path)
                extension_id = lineage.get("extension_dataset_id")
                if extension_id:
                    extension_id = self._validate_id(extension_id, "segment_id")
                    segment = (self.data_root / "segments" / extension_id).resolve()
                    if segment.exists():
                        segment_paths.add(segment)

        campaigns_used_elsewhere = {
            campaign
            for current_id, campaign in campaign_by_dataset.items()
            if current_id not in target_ids
        }
        campaign_paths = set()
        for current_id in target_ids:
            campaign = campaign_by_dataset.get(current_id)
            if not campaign or campaign in campaigns_used_elsewhere:
                continue
            user_id, campaign_id = campaign
            path = (
                self.data_root
                / "users"
                / user_id
                / "campaigns"
                / campaign_id
            ).resolve()
            if path.exists():
                campaign_paths.add(path)

        analysis_paths = set()
        analysis_run_count = 0
        for current_id in target_ids:
            path = (self.data_root / "analysis_runs" / current_id).resolve()
            if path.exists():
                analysis_paths.add(path)
                analysis_run_count += sum(item.is_dir() for item in path.iterdir())

        export_paths = set()
        exports_root = self.data_root / "exports"
        if exports_root.exists():
            for path in exports_root.glob("*.zip"):
                if any(
                    path.name == f"simulation_{current_id}.zip"
                    or path.name.startswith(f"analysis_{current_id}_")
                    for current_id in target_ids
                ):
                    export_paths.add(path.resolve())

        snapshot_paths = {dataset_roots[item] for item in target_ids}
        paths = (
            snapshot_paths
            | analysis_paths
            | segment_paths
            | campaign_paths
            | export_paths
        )
        return {
            "dataset_ids": tuple(sorted(target_ids)),
            "paths": tuple(sorted(paths, key=lambda item: len(item.parts), reverse=True)),
            "analysis_run_count": analysis_run_count,
            "segment_count": len(segment_paths),
            "campaign_count": len(campaign_paths),
            "export_count": len(export_paths),
            "total_bytes": sum(self._path_size(path) for path in paths),
        }

    def dataset_deletion_preview(self, dataset_id):
        scope = self._deletion_scope(dataset_id)
        return DatasetDeletionPreview(
            requested_dataset_id=self._validate_id(dataset_id, "dataset_id"),
            dataset_ids=scope["dataset_ids"],
            analysis_run_count=scope["analysis_run_count"],
            segment_count=scope["segment_count"],
            campaign_count=scope["campaign_count"],
            export_count=scope["export_count"],
            total_bytes=scope["total_bytes"],
        )

    def _remove_data_path(self, path):
        path = Path(path).resolve()
        if not self._is_relative_to(path, self.data_root.resolve()):
            raise ValueError("A exclusão deve permanecer dentro do diretório data.")
        for attempt in range(6):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.15 * (attempt + 1))

    def delete_dataset(self, dataset_id, confirmation):
        dataset_id = self._validate_id(dataset_id, "dataset_id")
        if str(confirmation).strip() != dataset_id:
            raise ValueError("A confirmação deve ser exatamente o ID do dataset.")
        scope = self._deletion_scope(dataset_id)
        for path in scope["paths"]:
            self._remove_data_path(path)
        return DatasetDeletionResult(
            deleted_dataset_ids=scope["dataset_ids"],
            deleted_path_count=len(scope["paths"]),
            reclaimed_bytes=scope["total_bytes"],
        )

    def list_analysis_runs(self, dataset_id=None):
        analysis_root = self.data_root / "analysis_runs"
        if not analysis_root.exists():
            return []
        selected_dataset = (
            self._validate_id(dataset_id, "dataset_id")
            if dataset_id is not None
            else None
        )
        records = []
        for run_root in analysis_root.glob("*/*"):
            if not run_root.is_dir():
                continue
            current_dataset = run_root.parent.name
            run_id = run_root.name
            if selected_dataset and current_dataset != selected_dataset:
                continue
            try:
                self._validate_id(current_dataset, "dataset_id")
                self._validate_id(run_id, "run_id")
            except ValueError:
                continue
            summary_path = run_root / "summary.json"
            evaluation_path = run_root / "evaluation.json"
            summary = (
                DatasetPackage.read_json(summary_path)
                if summary_path.is_file()
                else {}
            )
            evaluation = (
                DatasetPackage.read_json(evaluation_path)
                if evaluation_path.is_file()
                else {}
            )
            records.append(
                {
                    "dataset_id": current_dataset,
                    "run_id": run_id,
                    "finished_at": summary.get("finished_at"),
                    "alert_count": summary.get("alert_count"),
                    "precision": evaluation.get("precision"),
                    "recall": evaluation.get("recall"),
                    "f1_score": evaluation.get("f1_score"),
                    "size_bytes": self._path_size(run_root),
                }
            )
        return sorted(
            records,
            key=lambda item: (
                item.get("finished_at") or "",
                item["dataset_id"],
                item["run_id"],
            ),
            reverse=True,
        )

    def _analysis_deletion_scope(self, dataset_id, run_id):
        dataset_id = self._validate_id(dataset_id, "dataset_id")
        run_id = self._validate_id(run_id, "run_id")
        analysis_root = (self.data_root / "analysis_runs").resolve()
        run_root = (analysis_root / dataset_id / run_id).resolve()
        if not self._is_relative_to(run_root, analysis_root) or not run_root.is_dir():
            raise FileNotFoundError(
                f"Análise inexistente: {dataset_id}/{run_id}"
            )
        export_paths = set()
        exports_root = self.data_root / "exports"
        expected_name = f"analysis_{dataset_id}_{run_id}.zip"
        export_path = (exports_root / expected_name).resolve()
        if export_path.is_file():
            export_paths.add(export_path)
        paths = {run_root} | export_paths
        return {
            "dataset_id": dataset_id,
            "run_id": run_id,
            "run_root": run_root,
            "paths": tuple(paths),
            "output_bytes": self._path_size(run_root),
            "export_count": len(export_paths),
            "total_bytes": sum(self._path_size(path) for path in paths),
        }

    def analysis_deletion_preview(self, dataset_id, run_id):
        scope = self._analysis_deletion_scope(dataset_id, run_id)
        return AnalysisDeletionPreview(
            dataset_id=scope["dataset_id"],
            run_id=scope["run_id"],
            output_bytes=scope["output_bytes"],
            export_count=scope["export_count"],
            total_bytes=scope["total_bytes"],
        )

    def delete_analysis(self, dataset_id, run_id, confirmation):
        run_id = self._validate_id(run_id, "run_id")
        if str(confirmation).strip() != run_id:
            raise ValueError("A confirmação deve ser exatamente o run_id da análise.")
        scope = self._analysis_deletion_scope(dataset_id, run_id)
        for path in scope["paths"]:
            self._remove_data_path(path)
        dataset_directory = scope["run_root"].parent
        if dataset_directory.is_dir() and not any(dataset_directory.iterdir()):
            self._remove_data_path(dataset_directory)
        return AnalysisDeletionResult(
            dataset_id=scope["dataset_id"],
            run_id=scope["run_id"],
            deleted_path_count=len(scope["paths"]),
            reclaimed_bytes=scope["total_bytes"],
        )

    def run_coordination(self, dataset_id, run_id):
        scope = self._analysis_deletion_scope(dataset_id, run_id)
        coordinator_config_path = self.project_root / "config" / "coordinator_config.json"
        with open(coordinator_config_path, "r", encoding="utf-8") as file:
            coordinator_config = json.load(file)
        report = CoordinatorAgent(scope["run_root"], coordinator_config).run()
        return CoordinationRunResult(
            dataset_id=scope["dataset_id"],
            run_id=scope["run_id"],
            analysis_dir=scope["run_root"],
            report_path=scope["run_root"] / CoordinatorAgent.REPORT_NAME,
            decision=report["decision"],
            decision_label=report["decision_label"],
        )

    @staticmethod
    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def run_analysis(self, dataset_id, run_id=None, analysis_model_id=None):
        observable_manifest, evaluation_manifest = self.snapshot_paths(dataset_id)
        analysis_model_id = analysis_model_id or "modelo_regras_estatistico"
        analysis_model_id = self._validate_id(
            analysis_model_id, "analysis_model_id"
        )
        if analysis_model_id not in self.ANALYSIS_MODELS:
            raise ValueError("Modelo de analise nao reconhecido.")
        run_id = self._validate_id(
            run_id or self._analysis_run_id(analysis_model_id),
            "run_id",
        )
        analysis_config_path = self.project_root / "config" / "analysis_config.json"
        with open(analysis_config_path, "r", encoding="utf-8") as file:
            analysis_config = json.load(file)

        observable = ObservableDatasetPackage(observable_manifest)
        agent = AnalyticalAgent(observable, analysis_config, run_id)
        agent.run()

        evaluation_dataset = EvaluationDatasetPackage(evaluation_manifest)
        alerts = DatasetPackage.read_json(agent.output_dir / "alerts.json")
        observations = [
            DatasetPackage.read_json(path)
            for path in sorted((agent.output_dir / "observations").glob("frame_*.json"))
        ]
        evaluation = Evaluator(evaluation_dataset, analysis_config).evaluate(
            alerts, observations
        )
        evaluation_path = agent.output_dir / "evaluation.json"
        self._write_json(evaluation_path, evaluation)
        summary_path = agent.output_dir / "summary.json"
        summary = DatasetPackage.read_json(summary_path)
        summary["analysis_model"] = {
            "id": analysis_model_id,
            **self.ANALYSIS_MODELS[analysis_model_id],
        }
        summary["evaluation"] = evaluation
        summary["evaluation_status"] = "completed_by_external_evaluator"
        self._write_json(summary_path, summary)
        return AnalysisRunResult(
            dataset_id=dataset_id,
            run_id=run_id,
            output_dir=agent.output_dir,
            summary_path=summary_path,
            alerts_path=agent.output_dir / "alerts.json",
            evaluation_path=evaluation_path,
        )

    def create_zip(self, directory, archive_name):
        directory = Path(directory).resolve()
        archive_name = self._validate_id(archive_name, "archive_name")
        allowed_roots = [
            (self.data_root / "datasets").resolve(),
            (self.data_root / "analysis_runs").resolve(),
        ]
        if not any(
            self._is_relative_to(directory, allowed_root)
            for allowed_root in allowed_roots
        ):
            raise ValueError("Somente datasets e análises podem ser exportados.")
        exports = self.data_root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        target = exports / archive_name
        archive = shutil.make_archive(str(target), "zip", directory)
        return Path(archive)

    @staticmethod
    def evaluation_alert_rows(alerts, evaluation):
        """Une alertas produzidos e ocorrências não detectadas para exibição."""

        alerts_by_id = {item["alert_id"]: item for item in alerts}
        matched_alert_ids = set()
        rows = []
        for occurrence in evaluation.get("occurrences", []):
            matched_id = occurrence.get("matched_alert_id")
            alert = alerts_by_id.get(matched_id)
            if alert:
                matched_alert_ids.add(matched_id)
                rows.append(
                    {
                        "status": "Detectado",
                        "alerta": alert["alert_id"],
                        "ocorrência": occurrence["id"],
                        "tipo": alert["type"],
                        "animal": alert["track_id"],
                        "prioridade": alert["priority"],
                        "abertura": alert["opened_frame"],
                        "último frame": alert["last_frame"],
                        "confiança": alert["maximum_confidence"],
                    }
                )
            else:
                rows.append(
                    {
                        "status": "Não detectado",
                        "alerta": "—",
                        "ocorrência": occurrence["id"],
                        "tipo": occurrence["type"],
                        "animal": occurrence["animal_id"],
                        "prioridade": "critical",
                        "abertura": occurrence["start_frame"],
                        "último frame": occurrence["end_frame"],
                        "confiança": None,
                    }
                )

        false_positive_ids = set(evaluation.get("false_positive_alert_ids", []))
        for alert in alerts:
            if alert["alert_id"] in matched_alert_ids:
                continue
            rows.append(
                {
                    "status": (
                        "Falso positivo"
                        if alert["alert_id"] in false_positive_ids
                        else "Sem correspondência"
                    ),
                    "alerta": alert["alert_id"],
                    "ocorrência": "—",
                    "tipo": alert["type"],
                    "animal": alert["track_id"],
                    "prioridade": alert["priority"],
                    "abertura": alert["opened_frame"],
                    "último frame": alert["last_frame"],
                    "confiança": alert["maximum_confidence"],
                }
            )
        return sorted(rows, key=lambda item: (item["abertura"], item["tipo"]))

    @staticmethod
    def _is_relative_to(path, root):
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def result_as_dict(result):
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(result).items()
        }
