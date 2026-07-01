import argparse
import json
from pathlib import Path

from src.analytical.analytical_agent import AnalyticalAgent, PROJECT_ROOT
from src.analytical.dataset_exporter import DatasetSnapshotExporter
from src.analytical.dataset_package import (
    DatasetPackage,
    EvaluationDatasetPackage,
    ObservableDatasetPackage,
)
from src.analytical.evaluator import Evaluator


def load_analysis_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _manifest_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def analyze(args):
    manifest_path = _manifest_path(args.dataset)
    with open(manifest_path, "r", encoding="utf-8") as file:
        schema_version = json.load(file).get("schema_version")
    dataset = (
        ObservableDatasetPackage(manifest_path)
        if schema_version == 2
        else DatasetPackage(manifest_path)
    )
    config = load_analysis_config(args.config)
    agent = AnalyticalAgent(dataset, config, args.run_id)
    summary = agent.run()
    print("Análise concluída sem acesso ao ground truth.")
    print(f"Dataset: {summary['dataset_id']}")
    print(f"Frames: {summary['observation_count']}")
    print(f"Alertas: {summary['alert_count']}")
    print(f"Saída: {agent.output_dir}")


def package(args):
    output = args.output or f"config/datasets/{args.dataset_id}.json"
    path = DatasetPackage.create_manifest(
        args.dataset_id,
        args.source,
        args.start,
        args.end,
        output,
        args.description,
    )
    DatasetPackage(path)
    print(f"Manifesto legado criado: {path}")


def package_v2(args):
    destination = DatasetSnapshotExporter(
        args.source,
        args.dataset_id,
        args.start,
        args.end,
        include_raw=args.include_raw,
        config_path=args.simulation_config,
    ).export()
    print(f"Snapshot criado: {destination}")
    print(f"Inferência: {destination / 'observable_manifest.json'}")
    print(f"Avaliação: {destination / 'evaluation_manifest.json'}")


def evaluate(args):
    dataset = EvaluationDatasetPackage(args.evaluation_manifest)
    config = load_analysis_config(args.config)
    analysis_dir = _manifest_path(args.analysis_dir).resolve()
    allowed_root = (PROJECT_ROOT / "data" / "analysis_runs").resolve()
    try:
        analysis_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("analysis_dir deve estar dentro de data/analysis_runs.") from exc

    alerts = DatasetPackage.read_json(analysis_dir / "alerts.json")
    observations = [
        DatasetPackage.read_json(path)
        for path in sorted((analysis_dir / "observations").glob("frame_*.json"))
    ]
    result = Evaluator(dataset, config).evaluate(alerts, observations)
    evaluation_path = analysis_dir / "evaluation.json"
    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    summary_path = analysis_dir / "summary.json"
    if summary_path.is_file():
        summary = DatasetPackage.read_json(summary_path)
        summary["evaluation"] = result
        summary["evaluation_status"] = "completed_by_external_evaluator"
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Avaliação concluída: {evaluation_path}")


def list_datasets(args):
    catalog = _manifest_path(args.catalog)
    manifests = sorted(catalog.glob("*.json"))
    if not manifests:
        print("Nenhum dataset registrado.")
        return
    for manifest_path in manifests:
        dataset = DatasetPackage(manifest_path)
        print(
            f"{dataset.dataset_id}: frames {dataset.start_frame}-"
            f"{dataset.end_frame} | {dataset.source_root}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Agente analítico da fazenda.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analisa um dataset.")
    analyze_parser.add_argument(
        "--dataset",
        default="config/datasets/current_simulation.json",
        help="Manifesto observável v2 ou manifesto legado.",
    )
    analyze_parser.add_argument(
        "--config", default="config/analysis_config.json"
    )
    analyze_parser.add_argument("--run-id")
    analyze_parser.set_defaults(handler=analyze)

    package_parser = subparsers.add_parser(
        "package", help="Cria somente um manifesto legado por referência."
    )
    package_parser.add_argument("--dataset-id", required=True)
    package_parser.add_argument("--source", required=True)
    package_parser.add_argument("--start", type=int, required=True)
    package_parser.add_argument("--end", type=int, required=True)
    package_parser.add_argument("--output")
    package_parser.add_argument("--description", default="")
    package_parser.set_defaults(handler=package)

    package_v2_parser = subparsers.add_parser(
        "package-v2", help="Cria snapshot imutável e separado."
    )
    package_v2_parser.add_argument("--dataset-id", required=True)
    package_v2_parser.add_argument("--source", required=True)
    package_v2_parser.add_argument("--start", type=int, required=True)
    package_v2_parser.add_argument("--end", type=int, required=True)
    package_v2_parser.add_argument("--include-raw", action="store_true")
    package_v2_parser.add_argument(
        "--simulation-config", default="config/simulation_config.json"
    )
    package_v2_parser.set_defaults(handler=package_v2)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Avalia uma análise pronta usando o pacote privado."
    )
    evaluate_parser.add_argument("--evaluation-manifest", required=True)
    evaluate_parser.add_argument("--analysis-dir", required=True)
    evaluate_parser.add_argument(
        "--config", default="config/analysis_config.json"
    )
    evaluate_parser.set_defaults(handler=evaluate)

    list_parser = subparsers.add_parser("list", help="Lista datasets legados.")
    list_parser.add_argument("--catalog", default="config/datasets")
    list_parser.set_defaults(handler=list_datasets)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
