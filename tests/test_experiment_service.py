import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.services.experiment_service import (
    ContinuationRequest,
    ExperimentService,
    SpecialistSimulationRequest,
)
from src.simulator.scenario_validator import ScenarioValidator


class ExperimentServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ExperimentService()

    def test_build_config_keeps_herd_consistent(self):
        request = SpecialistSimulationRequest(
            campaign_id="teste_web",
            num_days=3,
            num_cattle=23,
            seed=99,
            temperature_offset_c=4,
            event_types=("extreme_heat", "animal_fallen"),
        )
        config = self.service.build_config(request)

        categories = config["nutrition"]["categories"]
        self.assertEqual(sum(item["count"] for item in categories.values()), 23)
        self.assertEqual(
            config["social_behavior"]["lots"][0]["animal_ids"], "1-23"
        )
        self.assertEqual(config["simulation"]["num_days"], 3)
        self.assertEqual(config["weather"]["temperature_by_hour"]["7"], 12)
        self.assertEqual(
            {item["type"] for item in config["events"]["schedule"]},
            {"extreme_heat"},
        )

    def test_disables_events_when_none_are_selected(self):
        config = self.service.build_config(
            SpecialistSimulationRequest(campaign_id="sem_eventos")
        )
        self.assertFalse(config["events"]["enabled"])
        self.assertEqual(config["events"]["schedule"], [])

    def test_random_fallen_event_has_full_observation_window(self):
        config = self.service.build_config(
            SpecialistSimulationRequest(campaign_id="fallen_window")
        )
        request = ContinuationRequest(
            base_dataset_id="base",
            new_dataset_id="extension",
            additional_days=3,
            target_active_animals=40,
            seed=17,
            event_types=("animal_fallen",),
            random_event_count=12,
        )
        events = self.service._random_future_events(
            config,
            {
                "completed_day": 0,
                "global_frame": 0,
                "animal_catalog": {
                    str(item): {"category": "vaca_adulta"}
                    for item in range(1, 41)
                },
            },
            request,
            list(range(1, 41)),
            41,
        )
        frame_hours = config["simulation"]["frame_hours"]
        for event in events:
            duration = int(event["duration_hours"])
            observed = sum(
                event["hour"] <= hour < event["hour"] + duration
                for hour in frame_hours
            )
            self.assertEqual(observed, duration)

    def test_rejects_unsafe_identifier(self):
        with self.assertRaises(ValueError):
            self.service.build_config(
                SpecialistSimulationRequest(campaign_id="../../segredo")
            )

    def test_evaluation_table_includes_missed_occurrence(self):
        rows = self.service.evaluation_alert_rows(
            alerts=[],
            evaluation={
                "occurrences": [
                    {
                        "id": "OCC_01",
                        "type": "animal_missing",
                        "animal_id": 33,
                        "start_frame": 100,
                        "end_frame": 120,
                        "detected": False,
                        "matched_alert_id": None,
                    }
                ],
                "false_positive_alert_ids": [],
            },
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "Não detectado")
        self.assertEqual(rows[0]["animal"], 33)

    def test_lifecycle_events_are_remapped_to_current_herd(self):
        config = self.service.build_config(
            SpecialistSimulationRequest(
                campaign_id="eventos_ciclo_vida",
                num_days=8,
                num_cattle=8,
                event_types=("animal_parturition", "animal_death"),
            )
        )
        events = {item["type"]: item for item in config["events"]["schedule"]}
        self.assertEqual(events["animal_parturition"]["animal_id"], 3)
        self.assertEqual(events["animal_parturition"]["parameters"]["calf_id"], 9)
        self.assertEqual(events["animal_death"]["animal_id"], 5)

    def test_branch_event_accepts_animal_after_administrative_entry(self):
        config = self.service.build_config(
            SpecialistSimulationRequest(
                campaign_id="branch_entry_event",
                num_cattle=8,
            )
        )
        config["campaign"]["mode"] = "branch"
        config["campaign"]["source_campaign_id"] = "source"
        config["administrative_events"] = {
            "enabled": True,
            "schedule": [
                {
                    "id": "ENTRY_9",
                    "type": "animal_entry",
                    "day": 1,
                    "hour": 7,
                    "animals": [{"id": 9}],
                }
            ],
        }
        config["events"] = {
            "enabled": True,
            "schedule": [
                {
                    "id": "MISSING_9",
                    "type": "animal_missing",
                    "day": 1,
                    "hour": 8,
                    "animal_id": 9,
                }
            ],
        }

        self.assertTrue(ScenarioValidator.validate(config))

        invalid = copy.deepcopy(config)
        invalid["events"]["schedule"][0]["hour"] = 7
        invalid["administrative_events"]["schedule"][0]["hour"] = 8
        with self.assertRaisesRegex(ValueError, "animal inexistente"):
            ScenarioValidator.validate(invalid)

    def test_branch_event_accepts_animal_restored_from_checkpoint(self):
        config = self.service.build_config(
            SpecialistSimulationRequest(
                campaign_id="branch_checkpoint_event",
                num_cattle=8,
            )
        )
        config["campaign"].update(
            {
                "mode": "branch",
                "source_campaign_id": "source",
                "checkpoint_animal_ids": list(range(1, 11)),
            }
        )
        config["events"] = {
            "enabled": True,
            "schedule": [
                {
                    "id": "MISSING_10",
                    "type": "animal_missing",
                    "day": 1,
                    "hour": 8,
                    "animal_id": 10,
                }
            ],
        }

        self.assertTrue(ScenarioValidator.validate(config))

    def test_dataset_deletion_cascades_descendants_and_derived_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = ExperimentService(root)

            def write_json(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            def dataset(dataset_id, campaign_id, parent=None, extension=None):
                target = root / "data" / "datasets" / dataset_id
                write_json(
                    target / "observable_manifest.json",
                    {
                        "schema_version": 2,
                        "dataset_id": dataset_id,
                        "frame_range": {"start": 1, "end": 1},
                    },
                )
                write_json(target / "evaluation_manifest.json", {})
                write_json(
                    target / "provenance/config/config_snapshot.json",
                    {
                        "campaign": {
                            "user_id": "web",
                            "campaign_id": campaign_id,
                        }
                    },
                )
                if parent:
                    write_json(
                        target / "provenance/lineage.json",
                        {
                            "parent_dataset_id": parent,
                            "extension_dataset_id": extension,
                        },
                    )

            dataset("base", "campaign_base")
            dataset("child", "campaign_child", "base", "segment_child")
            dataset("unrelated", "campaign_base")
            for campaign in ("campaign_base", "campaign_child"):
                path = root / "data/users/web/campaigns" / campaign / "marker.txt"
                path.parent.mkdir(parents=True)
                path.write_text("campaign", encoding="utf-8")
            segment = root / "data/segments/segment_child/marker.txt"
            segment.parent.mkdir(parents=True)
            segment.write_text("segment", encoding="utf-8")
            analysis = root / "data/analysis_runs/child/run_1/result.json"
            write_json(analysis, {})
            export = root / "data/exports/analysis_child_run_1.zip"
            export.parent.mkdir(parents=True)
            export.write_bytes(b"zip")

            preview = service.dataset_deletion_preview("base")
            self.assertEqual(set(preview.dataset_ids), {"base", "child"})
            self.assertEqual(preview.analysis_run_count, 1)
            with self.assertRaises(ValueError):
                service.delete_dataset("base", "incorreto")

            result = service.delete_dataset("base", "base")

            self.assertEqual(set(result.deleted_dataset_ids), {"base", "child"})
            self.assertFalse((root / "data/datasets/base").exists())
            self.assertFalse((root / "data/datasets/child").exists())
            self.assertFalse((root / "data/segments/segment_child").exists())
            self.assertFalse((root / "data/analysis_runs/child").exists())
            self.assertFalse(export.exists())
            self.assertFalse(
                (root / "data/users/web/campaigns/campaign_child").exists()
            )
            self.assertTrue(
                (root / "data/users/web/campaigns/campaign_base").exists()
            )
            self.assertTrue((root / "data/datasets/unrelated").exists())

    def test_analysis_deletion_preserves_dataset_and_removes_its_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = ExperimentService(root)
            dataset = root / "data/datasets/experiment/observable_manifest.json"
            dataset.parent.mkdir(parents=True)
            dataset.write_text("{}", encoding="utf-8")
            run = root / "data/analysis_runs/experiment/run_old"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "finished_at": "2026-07-01T10:00:00",
                        "alert_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            (run / "evaluation.json").write_text(
                json.dumps({"precision": 0.4, "recall": 0.5, "f1_score": 0.44}),
                encoding="utf-8",
            )
            export = root / "data/exports/analysis_experiment_run_old.zip"
            export.parent.mkdir(parents=True)
            export.write_bytes(b"zip")

            records = service.list_analysis_runs("experiment")
            preview = service.analysis_deletion_preview("experiment", "run_old")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_id"], "run_old")
            self.assertEqual(preview.export_count, 1)
            with self.assertRaises(ValueError):
                service.delete_analysis("experiment", "run_old", "wrong")

            result = service.delete_analysis(
                "experiment", "run_old", "run_old"
            )

            self.assertEqual(result.run_id, "run_old")
            self.assertFalse(run.exists())
            self.assertFalse(export.exists())
            self.assertTrue(dataset.exists())


if __name__ == "__main__":
    unittest.main()
