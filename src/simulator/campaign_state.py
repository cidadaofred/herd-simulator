import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path


class CampaignStateStore:
    """Isola saídas e mantém checkpoints retomáveis de uma campanha."""

    SCHEMA_VERSION = 1
    SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, project_root: Path, config: dict):
        self.project_root = project_root
        self.config = config
        self.campaign = config.get("campaign", {})
        self.enabled = bool(self.campaign.get("enabled", False))
        self.user_id = self._safe_id(self.campaign.get("user_id", "local"), "user_id")
        self.campaign_id = self._safe_id(
            self.campaign.get("campaign_id", "fazenda_padrao"), "campaign_id"
        )
        self.mode = self.campaign.get("mode", "new")
        if self.mode not in {"new", "resume", "branch"}:
            raise ValueError("campaign.mode deve ser new, resume ou branch.")

        if self.enabled:
            self.root = (
                project_root
                / "data"
                / "users"
                / self.user_id
                / "campaigns"
                / self.campaign_id
            )
        else:
            self.root = project_root / "data"
        self.checkpoints_dir = self.root / "checkpoints"
        self.runs_dir = self.root / "runs"
        self.manifest_path = self.root / "campaign_manifest.json"
        self.environment_signature = self._environment_signature(config)

    @classmethod
    def _safe_id(cls, value, field):
        value = str(value)
        if not cls.SAFE_ID.fullmatch(value):
            raise ValueError(f"campaign.{field} contém caracteres inválidos.")
        return value

    @staticmethod
    def _environment_signature(config):
        def without_documentation(value):
            if isinstance(value, dict):
                return {
                    key: without_documentation(item)
                    for key, item in value.items()
                    if not key.startswith("_") and key != "description"
                }
            if isinstance(value, list):
                return [without_documentation(item) for item in value]
            return value

        immutable = {
            "farm": without_documentation(config.get("farm", {})),
            "environment": without_documentation(config.get("environment", {})),
        }
        payload = json.dumps(
            immutable, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _write_json(path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        temporary.replace(path)

    def output_directories(self):
        return {
            "generated_frames": self.root / "generated_frames",
            "base_map": self.root / "base_map",
            "metadata": self.root / "metadata",
            "day_summaries": self.root / "day_summaries",
            "environment_state": self.root / "environment_state",
            "occurrence_ground_truth": self.root / "occurrence_ground_truth",
            "administrative_events": self.root / "administrative_events",
        }

    def _campaign_root(self, campaign_id):
        campaign_id = self._safe_id(campaign_id, "source_campaign_id")
        return (
            self.project_root
            / "data"
            / "users"
            / self.user_id
            / "campaigns"
            / campaign_id
        )

    @staticmethod
    def _latest_checkpoint(checkpoints_dir: Path):
        latest = checkpoints_dir / "latest.json"
        if latest.exists():
            return latest
        candidates = sorted(checkpoints_dir.glob("checkpoint_day_*.json"))
        if not candidates:
            raise ValueError(f"Nenhum checkpoint encontrado em {checkpoints_dir}.")
        return candidates[-1]

    def _checkpoint_to_load(self):
        if not self.enabled or self.mode == "new":
            return None
        if self.mode == "resume":
            source_root = self.root
        else:
            source_id = self.campaign.get("source_campaign_id")
            if not source_id:
                raise ValueError("campaign.source_campaign_id é obrigatório no modo branch.")
            source_root = self._campaign_root(source_id)
        requested = self.campaign.get("source_checkpoint")
        if requested:
            requested = Path(str(requested)).name
            checkpoint = source_root / "checkpoints" / requested
            if not checkpoint.exists():
                raise ValueError(f"Checkpoint não encontrado: {checkpoint}.")
            return checkpoint
        return self._latest_checkpoint(source_root / "checkpoints")

    def prepare(self):
        for directory in self.output_directories().values():
            directory.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.runs_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self._checkpoint_to_load()
        if self.enabled and self.mode == "new" and self.manifest_path.exists():
            if not self.campaign.get("allow_overwrite", False):
                raise ValueError(
                    f"A campanha {self.campaign_id} já existe. Use mode=resume ou outro campaign_id."
                )
            raise ValueError(
                "allow_overwrite não remove uma campanha existente por segurança; use outro campaign_id."
            )
        if self.enabled and self.mode == "branch" and self.manifest_path.exists():
            raise ValueError(
                f"A campanha de destino {self.campaign_id} já existe; escolha outro campaign_id."
            )

        state = None
        lineage = None
        if checkpoint_path:
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                state = json.load(file)
            if state.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("Versão de checkpoint incompatível.")
            if state.get("environment_signature") != self.environment_signature:
                raise ValueError(
                    "A geometria/recursos da fazenda diferem do checkpoint. "
                    "Continue com o mesmo ambiente ou crie uma campanha nova."
                )
            lineage = {
                "source_campaign_id": state.get("campaign_id"),
                "source_checkpoint": checkpoint_path.name,
                "source_completed_day": state.get("completed_day"),
            }

        if self.enabled and self.mode in {"new", "branch"}:
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "user_id": self.user_id,
                "campaign_id": self.campaign_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "start_date": self.campaign.get("start_date", "2026-06-20"),
                "environment_signature": self.environment_signature,
                "lineage": lineage,
                "config_snapshot": self.config,
            }
            self._write_json(self.manifest_path, manifest)
        if self.enabled:
            self.run_id = uuid.uuid4().hex
            self.run_path = self.runs_dir / f"run_{self.run_id}.json"
            self._write_json(
                self.run_path,
                {
                    "run_id": self.run_id,
                    "mode": self.mode,
                    "status": "started",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "requested_days": int(self.config["simulation"]["num_days"]),
                    "source_checkpoint": checkpoint_path.name if checkpoint_path else None,
                    "config_snapshot": self.config,
                },
            )
        return state

    def finish_run(self, completed_day: int, global_frame: int):
        if not self.enabled:
            return
        with open(self.run_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        payload.update(
            {
                "status": "completed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "completed_day": completed_day,
                "global_frame": global_frame,
            }
        )
        self._write_json(self.run_path, payload)

    def save_checkpoint(self, completed_day: int, state: dict):
        if not self.enabled:
            return None
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "user_id": self.user_id,
            "campaign_id": self.campaign_id,
            "environment_signature": self.environment_signature,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "completed_day": completed_day,
            **state,
        }
        checkpoint = self.checkpoints_dir / f"checkpoint_day_{completed_day:06d}.json"
        self._write_json(checkpoint, payload)
        self._write_json(self.checkpoints_dir / "latest.json", payload)
        return checkpoint
