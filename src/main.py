import json
import sys
from pathlib import Path


# src/main.py
# sobe dois níveis:
# main.py -> src -> raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# adiciona src ao PYTHONPATH
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.simulator_agent import SimulatorAgent


def _deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return override


def load_config(config_override=None):
    config_path = PROJECT_ROOT / "config" / "simulation_config.json"
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if config_override:
        override_path = Path(config_override)
        if not override_path.is_absolute():
            override_path = PROJECT_ROOT / override_path
        with open(override_path, "r", encoding="utf-8") as file:
            config = _deep_merge(config, json.load(file))
    return config


def main():

    config = load_config(sys.argv[1] if len(sys.argv) > 1 else None)

    simulator = SimulatorAgent(config)

    generated = simulator.run()

    print("Simulação concluída.")
    print(f"Frames gerados: {len(generated)}")

    print("\nArquivos gerados:")

    print(f"Campanha: {simulator.campaign_store.campaign_id}")
    print(f"Diretório: {simulator.campaign_store.root}")
    for directory in simulator.campaign_store.output_directories().values():
        print(directory)


if __name__ == "__main__":
    main()
