import json
from datetime import datetime
from pathlib import Path

from src.analytical.alert_builder import AlertBuilder
from src.analytical.observation_builder import ObservationBuilder
from src.analytical.rule_engine import RuleEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AnalyticalAgent:
    """Orquestra observação, inferência temporal e avaliação isolada."""

    def __init__(self, dataset, config, run_id=None):
        self.dataset = dataset
        self.config = config
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = (
            PROJECT_ROOT
            / "data"
            / "analysis_runs"
            / dataset.dataset_id
            / self.run_id
        )
        if self.output_dir.exists():
            raise ValueError(f"A execução analítica {self.run_id} já existe.")
        self.output_dir.mkdir(parents=True)

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    @staticmethod
    def _narrative(prediction):
        if prediction["type"] == "animal_fallen":
            evidence = prediction["evidence"]
            return (
                f"O track {prediction['track_id']} permaneceu praticamente imóvel "
                f"por {evidence['stationary_window_frames']} frames, com amplitude "
                f"de {evidence['position_span_px']} px e isolamento de "
                f"{evidence['nearest_neighbor_px']} px."
            )
        if prediction["type"] == "animal_death":
            evidence = prediction["evidence"]
            return (
                f"O track {prediction['track_id']} permaneceu imóvel por "
                f"{evidence['stationary_span_frames']} frames e apresentou "
                f"mudança visual RGB de {evidence['appearance_change_rgb']}."
            )
        if prediction["type"] == "animal_parturition":
            evidence = prediction["evidence"]
            return (
                f"Após redução de movimento do track {prediction['track_id']}, "
                f"o novo track {evidence['calf_track_id']} permaneceu próximo "
                f"por {evidence['proximity_persistence_frames']} frames."
            )
        evidence = prediction["evidence"]
        return (
            f"O track {prediction['track_id']} não foi observado por "
            f"{evidence['consecutive_reliable_absences']} frames confiáveis; "
            f"{evidence['observed_count']} de {evidence['reference_track_count']} "
            "tracks conhecidos foram observados no frame atual."
        )

    def run(self):
        started_at = datetime.now().isoformat(timespec="seconds")
        observations = ObservationBuilder(self.dataset, self.output_dir).build()
        engine = RuleEngine(self.config)
        predictions = engine.analyze(observations)
        for observation in observations:
            self._write_json(
                self.output_dir
                / "observations"
                / f"frame_{int(observation['frame']):04d}.json",
                observation,
            )
        self._write_json(
            self.output_dir / "population_history.json",
            engine.population_history,
        )
        for index, prediction in enumerate(predictions, start=1):
            prediction["prediction_id"] = f"PRED_{index:05d}"
            prediction["status"] = "candidate"
            prediction["narrative"] = self._narrative(prediction)
        self._write_json(self.output_dir / "predictions.json", predictions)

        alerts, alert_history = AlertBuilder(self.config).build(predictions)
        self._write_json(self.output_dir / "alerts.json", alerts)
        self._write_json(self.output_dir / "alert_history.json", alert_history)

        summary = {
            "dataset_id": self.dataset.dataset_id,
            "run_id": self.run_id,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "frame_range": {
                "start": self.dataset.start_frame,
                "end": self.dataset.end_frame,
            },
            "observation_count": len(observations),
            "prediction_count": len(predictions),
            "alert_count": len(alerts),
            "final_observed_population_estimate": (
                len(engine.reference_tracks)
            ),
            "observed_population_change_count": max(
                0, len(engine.population_history) - 1
            ),
            "prediction_distribution": {
                event_type: sum(
                    prediction["type"] == event_type for prediction in predictions
                )
                for event_type in {prediction["type"] for prediction in predictions}
            },
            "evaluation": None,
            "evaluation_status": "pending_external_evaluator",
            "ground_truth_used_by_agent": False,
        }
        self._write_json(self.output_dir / "summary.json", summary)
        self._write_json(
            self.output_dir / "analysis_manifest.json",
            {
                "dataset_manifest": str(self.dataset.manifest_path),
                "dataset_id": self.dataset.dataset_id,
                "run_id": self.run_id,
                "analysis_config": self.config,
                "outputs": {
                    "observations": "observations/",
                    "predictions": "predictions.json",
                    "alerts": "alerts.json",
                    "alert_history": "alert_history.json",
                    "population_history": "population_history.json",
                    "evaluation": None,
                    "summary": "summary.json",
                },
            },
        )
        return summary
