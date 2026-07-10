import json
from datetime import datetime
from pathlib import Path


class CoordinatorAgent:
    """Audita uma analise em modo experimental pos-avaliacao."""

    REPORT_NAME = "coordination_report.json"

    OPERATIONAL_PROXY = "operational_proxy"
    POST_EVALUATION_GROUND_TRUTH = "post_evaluation_ground_truth"

    def __init__(self, analysis_dir, config):
        self.analysis_dir = Path(analysis_dir).resolve()
        self.config = config

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
    def _ratio(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else 0.0

    @staticmethod
    def _group_checks(checks):
        grouped = {
            CoordinatorAgent.OPERATIONAL_PROXY: [],
            CoordinatorAgent.POST_EVALUATION_GROUND_TRUTH: [],
        }
        for check in checks:
            grouped.setdefault(check.get("category", "other"), []).append(check)
        return grouped

    def _load_inputs(self):
        paths = {
            "summary": self.analysis_dir / "summary.json",
            "evaluation": self.analysis_dir / "evaluation.json",
            "alerts": self.analysis_dir / "alerts.json",
            "manifest": self.analysis_dir / "analysis_manifest.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        payloads = {
            name: self._read_json(path)
            for name, path in paths.items()
            if path.is_file()
        }
        return paths, missing, payloads

    def _metric_checks(self, evaluation):
        gate = self.config.get("quality_gate", {})
        precision = float(evaluation.get("precision", 0.0))
        recall = float(evaluation.get("recall", 0.0))
        f1_score = float(evaluation.get("f1_score", 0.0))
        alert_count = int(evaluation.get("alert_count", 0))
        occurrence_count = int(evaluation.get("occurrence_count", 0))
        false_positives = int(evaluation.get("false_positives", 0))
        false_negatives = int(evaluation.get("false_negatives", 0))
        false_positive_rate = self._ratio(false_positives, alert_count)

        no_positive_case = occurrence_count == 0 and alert_count == 0
        checks = [
            {
                "name": "precision",
                "category": self.POST_EVALUATION_GROUND_TRUTH,
                "observed": precision,
                "threshold": float(gate.get("minimum_precision", 0.0)),
                "passed": no_positive_case
                or precision >= float(gate.get("minimum_precision", 0.0)),
            },
            {
                "name": "recall",
                "category": self.POST_EVALUATION_GROUND_TRUTH,
                "observed": recall,
                "threshold": float(gate.get("minimum_recall", 0.0)),
                "passed": no_positive_case
                or recall >= float(gate.get("minimum_recall", 0.0)),
            },
            {
                "name": "f1_score",
                "category": self.POST_EVALUATION_GROUND_TRUTH,
                "observed": f1_score,
                "threshold": float(gate.get("minimum_f1", 0.0)),
                "passed": no_positive_case
                or f1_score >= float(gate.get("minimum_f1", 0.0)),
            },
            {
                "name": "false_positive_rate",
                "category": self.POST_EVALUATION_GROUND_TRUTH,
                "observed": false_positive_rate,
                "threshold": float(gate.get("maximum_false_positive_rate", 1.0)),
                "passed": false_positive_rate
                <= float(gate.get("maximum_false_positive_rate", 1.0)),
                "direction": "maximum",
            },
        ]
        return checks, {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "alert_count": alert_count,
            "occurrence_count": occurrence_count,
            "no_positive_case": no_positive_case,
        }

    def _coverage_checks(self, evaluation, summary):
        gate = self.config.get("quality_gate", {})
        coverage = evaluation.get("coverage", {})
        frame_coverage = float(coverage.get("frame_coverage", 0.0))
        observation_count = int(summary.get("observation_count", 0))
        expected_frames = int(coverage.get("expected_frames", 0))
        analyzed_frames = int(coverage.get("analyzed_frames", 0))
        return [
            {
                "name": "frame_coverage",
                "category": self.OPERATIONAL_PROXY,
                "observed": frame_coverage,
                "threshold": float(gate.get("minimum_frame_coverage", 1.0)),
                "passed": frame_coverage
                >= float(gate.get("minimum_frame_coverage", 1.0)),
            },
            {
                "name": "observation_count_matches_coverage",
                "category": self.OPERATIONAL_PROXY,
                "observed": observation_count,
                "threshold": analyzed_frames,
                "passed": observation_count == analyzed_frames,
            },
            {
                "name": "expected_frames_processed",
                "category": self.OPERATIONAL_PROXY,
                "observed": analyzed_frames,
                "threshold": expected_frames,
                "passed": analyzed_frames == expected_frames,
            },
        ]

    @staticmethod
    def _issue_counts(evaluation):
        missed = evaluation.get("false_negative_occurrence_ids", [])
        false_positive_ids = evaluation.get("false_positive_alert_ids", [])
        occurrence_types = {}
        for occurrence in evaluation.get("occurrences", []):
            if occurrence.get("detected"):
                continue
            occurrence_type = occurrence.get("type", "unknown")
            occurrence_types[occurrence_type] = (
                occurrence_types.get(occurrence_type, 0) + 1
            )
        return {
            "false_positive_alert_ids": false_positive_ids,
            "false_negative_occurrence_ids": missed,
            "missed_occurrence_types": occurrence_types,
        }

    def _warnings(self, metrics, checks, issues):
        warnings = []
        if metrics["false_positives"]:
            warnings.append(
                f"{metrics['false_positives']} alerta(s) falso(s) positivo(s) foram identificados."
            )
        if metrics["false_negatives"]:
            warnings.append(
                f"{metrics['false_negatives']} ocorrencia(s) real(is) nao foram detectadas."
            )
        for check in checks:
            if not check["passed"]:
                warnings.append(
                    f"Criterio '{check['name']}' nao atingiu o limite configurado."
                )
        if issues["missed_occurrence_types"]:
            warnings.append(
                "Tipos de ocorrencia perdidos: "
                + ", ".join(
                    f"{event_type}={count}"
                    for event_type, count in sorted(
                        issues["missed_occurrence_types"].items()
                    )
                )
            )
        return warnings

    def _decision(self, metrics, checks, warnings):
        policy = self.config.get("decision_policy", {})
        hard_fail_checks = {
            "frame_coverage",
            "observation_count_matches_coverage",
            "expected_frames_processed",
            "ground_truth_not_used_by_analytical_agent",
        }
        if any(
            not check["passed"] and check["name"] in hard_fail_checks
            for check in checks
        ):
            return "rejected"

        if all(check["passed"] for check in checks) and not warnings:
            return "approved"

        warning_floor = float(policy.get("approve_with_warnings_if_f1_above", 0.0))
        if metrics["no_positive_case"] and metrics["false_positives"] == 0:
            return "approved"
        if float(metrics["f1_score"]) >= warning_floor:
            return "approved_with_warnings"
        return "rejected"

    def _base_report(self, now, paths):
        return {
            "coordinator_agent": "CoordinatorAgent",
            "coordination_version": self.config.get("version", "unknown"),
            "coordination_mode": self.config.get(
                "coordination_mode", "experimental_post_evaluation"
            ),
            "coordinated_at": now,
            "evaluation_method": self.config.get("evaluation_method"),
            "ground_truth_dependency": self.config.get(
                "ground_truth_dependency", {}
            ),
            "paths": {name: str(path) for name, path in paths.items()},
        }

    def run(self):
        paths, missing, payloads = self._load_inputs()
        now = datetime.now().isoformat(timespec="seconds")
        if missing:
            report = {
                **self._base_report(now, paths),
                "decision": "rejected",
                "decision_label": "Reprovado",
                "missing_inputs": missing,
                "warnings": [
                    "A coordenacao nao pode aprovar analises incompletas."
                ],
            }
            self._write_json(self.analysis_dir / self.REPORT_NAME, report)
            return report

        summary = payloads["summary"]
        evaluation = payloads["evaluation"]
        manifest = payloads["manifest"]

        metric_checks, metrics = self._metric_checks(evaluation)
        coverage_checks = self._coverage_checks(evaluation, summary)
        isolation_checks = [
            {
                "name": "ground_truth_not_used_by_analytical_agent",
                "category": self.OPERATIONAL_PROXY,
                "observed": bool(summary.get("ground_truth_used_by_agent")),
                "threshold": False,
                "passed": not bool(summary.get("ground_truth_used_by_agent")),
            }
        ]
        checks = metric_checks + coverage_checks + isolation_checks
        grouped_checks = self._group_checks(checks)
        issues = self._issue_counts(evaluation)
        warnings = self._warnings(metrics, checks, issues)
        decision = self._decision(metrics, checks, warnings)
        labels = {
            "approved": "Aprovado",
            "approved_with_warnings": "Aprovado com ressalvas",
            "rejected": "Reprovado",
        }
        report = {
            **self._base_report(now, paths),
            "dataset_id": summary.get("dataset_id") or evaluation.get("dataset_id"),
            "run_id": summary.get("run_id"),
            "decision": decision,
            "decision_label": labels[decision],
            "metrics_used": self.config.get("metrics_used", []),
            "observed_metrics": metrics,
            "checks": checks,
            "checks_by_dependency": grouped_checks,
            "warnings": warnings,
            "issues": issues,
            "recommended_use": self.config.get("recommended_use", {}).get(
                decision, ""
            ),
            "ground_truth_used_by_analytical_agent": bool(
                summary.get("ground_truth_used_by_agent")
            ),
            "analysis_manifest_method": {
                "analysis_config_keys": sorted(
                    manifest.get("analysis_config", {}).keys()
                ),
                "outputs": manifest.get("outputs", {}),
            },
        }
        self._write_json(self.analysis_dir / self.REPORT_NAME, report)
        return report
