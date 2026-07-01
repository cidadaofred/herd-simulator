import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from src.simulator.behavior_planner import BehaviorPlanner
from src.simulator.administrative_event_generator import AdministrativeEventGenerator
from src.simulator.campaign_state import CampaignStateStore
from src.simulator.event_generator import EventGenerator
from src.simulator.farm_environment import FarmEnvironment
from src.simulator.herd_nutrition import HerdNutritionModel
from src.simulator.herd_social import HerdSocialModel
from src.simulator.image_generator import ImageGenerator
from src.simulator.pasture_model import PastureModel
from src.simulator.scenario_validator import ScenarioValidator
from src.simulator.spatial_behavior import SpatialBehaviorModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SimulatorAgent:
    """Simula ambiente, nutrição e movimentos assíncronos do rebanho."""

    def __init__(self, config: dict):
        ScenarioValidator.validate(config)
        self.config = config
        self.sim_cfg = config["simulation"]
        self.move_cfg = config["movement"]
        self.weather_cfg = config["weather"]
        self.rng = random.Random(self.sim_cfg["seed"])
        self.campaign_store = CampaignStateStore(PROJECT_ROOT, config)

        farm_cfg = config["farm"]
        self.farm = FarmEnvironment(
            farm_cfg["width"],
            farm_cfg["height"],
            config["environment"],
        )
        self.pasture_model = PastureModel(self.farm, config["pasture"])
        self.behavior_planner = BehaviorPlanner(
            self.farm,
            config["pasture"],
            config["behavior_planner"],
        )
        self.event_generator = EventGenerator(config)
        self.administrative_event_generator = AdministrativeEventGenerator(config)
        self.num_cattle = self.sim_cfg["num_bovinos"]
        self.nutrition_model = HerdNutritionModel(
            config["nutrition"],
            self.num_cattle,
        )
        self.social_model = HerdSocialModel(
            config["social_behavior"],
            self.num_cattle,
        )
        self.social_model.configure_family_links(self.nutrition_model.animals)
        self.spatial_model = SpatialBehaviorModel(
            self.farm,
            config["spatial_behavior"],
            self.rng,
        )
        resume_state = self.campaign_store.prepare()

        output_dirs = self.campaign_store.output_directories()
        self.generated_frames_dir = output_dirs["generated_frames"]
        self.base_map_dir = output_dirs["base_map"]
        self.metadata_dir = output_dirs["metadata"]
        self.day_summaries_dir = output_dirs["day_summaries"]
        self.environment_state_dir = output_dirs["environment_state"]
        self.occurrence_ground_truth_dir = output_dirs["occurrence_ground_truth"]
        self.administrative_events_dir = output_dirs["administrative_events"]
        for directory in (
            self.generated_frames_dir,
            self.base_map_dir,
            self.metadata_dir,
            self.day_summaries_dir,
            self.environment_state_dir,
            self.occurrence_ground_truth_dir,
            self.administrative_events_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.image_generator = ImageGenerator(
            self.farm,
            self.generated_frames_dir,
            self.base_map_dir,
            self.pasture_model,
            config["rendering"],
        )
        self.animal_profiles = self._create_animal_profiles()
        self.animal_states = {
            cattle_id: "normal" for cattle_id in range(1, self.num_cattle + 1)
        }
        self.inventory_status = {
            cattle_id: (
                "active" if self.social_model.arrival_day(cattle_id) <= 1 else "pending"
            )
            for cattle_id in range(1, self.num_cattle + 1)
        }
        self.birth_records = {}
        self.death_records = {}
        self.missing_records = {}
        self.external_animal_records = {}
        initial_home = config["behavior_planner"]["initial_home_paddock"]
        self.herd_positions = self._initial_positions(initial_home, day=1)
        self.completed_day = 0
        self.global_frame = 0
        self.start_date = datetime.fromisoformat(
            config.get("campaign", {}).get("start_date", "2026-06-20")
        )
        if resume_state:
            self._restore_checkpoint(resume_state)

    def _create_animal_profiles(self):
        return {
            cattle_id: {
                "angle": self.rng.uniform(0, 2 * math.pi),
                "radius": self.rng.uniform(0.25, 1.0),
                "pace": self.rng.uniform(0.88, 1.12),
            }
            for cattle_id in range(1, self.num_cattle + 1)
        }

    def _shelter_center(self, paddock: str):
        return self.farm.refuge_center(paddock)

    def _can_use_shelter(self, cattle_id: int, paddock: str, day: int):
        capacity = self.farm.shelter_capacity(paddock)
        if capacity <= 0:
            return False
        rotating_rank = (cattle_id - 1 + day * 7) % self.num_cattle
        return rotating_rank < min(capacity, self.num_cattle)

    def _refuge_center_for_behavior(self, paddock: str, behavior: str):
        if behavior.startswith("sombra_natural"):
            return self.farm.natural_shadow_center(paddock)
        if behavior == "repouso_sem_abrigo":
            return self.farm.get_paddock_center(paddock)
        return self.farm.refuge_center(paddock)

    def _initial_positions(self, home_paddock: str, day: int):
        shelter = self.move_cfg["shelter"]
        center = self._shelter_center(home_paddock)
        positions = []
        for cattle_id in self.social_model.active_ids(day):
            paddock = self.social_model.preferred_shelter(
                cattle_id,
                home_paddock,
                day,
            )
            behavior = self._refuge_behavior(cattle_id, paddock, "rotina", day)
            center = self._refuge_center_for_behavior(paddock, behavior)
            x, y = self._profile_target(
                cattle_id,
                center,
                shelter["spread_x"],
                shelter["spread_y"],
                0.0,
            )
            x, y = self._clamp_to_paddock(x, y, paddock)
            animal = self.nutrition_model.animals[cattle_id]
            positions.append(
                {
                    "id": cattle_id,
                    "paddock": paddock,
                    "x": x,
                    "y": y,
                    "state": behavior,
                    "category": animal.category,
                    "cohort": animal.cohort,
                    "lot_id": self.social_model.lot_for(cattle_id),
                }
            )
        return positions

    def _activate_arrivals(self, day: int):
        arrivals = self.social_model.arrivals(day)
        if not arrivals:
            return []
        activated = []
        for cattle_id in arrivals:
            if self.inventory_status.get(cattle_id) == "sold":
                continue
            self.inventory_status[cattle_id] = "active"
            if self._add_animal_position(cattle_id, day):
                activated.append(cattle_id)
        return activated

    def _add_animal_position(self, cattle_id: int, day: int):
        existing_ids = {animal["id"] for animal in self.herd_positions}
        if cattle_id in existing_ids:
            return False
        paddock = self.social_model.preferred_shelter(
            cattle_id,
            self.config["behavior_planner"]["initial_home_paddock"],
            day,
        )
        behavior = self._refuge_behavior(cattle_id, paddock, "rotina", day)
        center = self._refuge_center_for_behavior(paddock, behavior)
        shelter = self.move_cfg["shelter"]
        x, y = self._profile_target(
            cattle_id,
            center,
            shelter["spread_x"],
            shelter["spread_y"],
            day * 0.11,
        )
        x, y = self._clamp_to_paddock(x, y, paddock)
        animal = self.nutrition_model.animals[cattle_id]
        self.herd_positions.append(
            {
                "id": cattle_id,
                "paddock": paddock,
                "x": x,
                "y": y,
                "state": behavior,
                "category": animal.category,
                "cohort": animal.cohort,
                "lot_id": self.social_model.lot_for(cattle_id),
            }
        )
        return True

    def _next_cattle_id(self):
        known = set(self.nutrition_model.animals) | set(self.inventory_status)
        return max(known, default=0) + 1

    def _register_newborn(self, record: dict):
        mother_id = int(record["mother_id"])
        calf_id = int(record["calf_id"])
        if calf_id in self.nutrition_model.animals:
            raise ValueError(f"O ID do terneiro {calf_id} já existe no inventário.")
        mother = next(
            (item for item in self.herd_positions if item["id"] == mother_id),
            None,
        )
        if mother is None:
            raise ValueError(f"A mãe {mother_id} não está visível no rebanho.")

        nutrition = self.nutrition_model.add_newborn(calf_id)
        self.social_model.add_newborn(calf_id, mother_id)
        self.num_cattle = max(self.num_cattle, calf_id)
        self.animal_profiles[calf_id] = {
            "angle": self.rng.uniform(0, 2 * math.pi),
            "radius": self.rng.uniform(0.25, 0.55),
            "pace": self.rng.uniform(0.95, 1.08),
        }
        self.animal_states[calf_id] = "normal"
        self.inventory_status[calf_id] = "active"
        offset_x = 8 if calf_id % 2 else -8
        offset_y = 6 if calf_id % 3 else -6
        x, y = self._clamp_to_paddock(
            mother["x"] + offset_x,
            mother["y"] + offset_y,
            mother["paddock"],
        )
        self.herd_positions.append(
            {
                "id": calf_id,
                "paddock": mother["paddock"],
                "x": x,
                "y": y,
                "state": "acompanha_mae",
                "physical_state": "normal",
                "category": nutrition.category,
                "cohort": nutrition.cohort,
                "lot_id": self.social_model.lot_for(calf_id),
                "mother_id": mother_id,
            }
        )
        record["born"] = True

    def _register_external_animal(self, specification: dict, day: int):
        cattle_id = int(specification["id"])
        if cattle_id in self.nutrition_model.animals:
            return False
        category = specification.get("category", "novilha")
        lot_id = specification.get("lot_id", f"lote_externo_dia_{day}")
        preferred_shelter = specification.get(
            "preferred_shelter",
            self.config["behavior_planner"]["initial_home_paddock"],
        )
        cohort = specification.get("cohort", f"{category}_externo_dia_{day}")
        self.nutrition_model.add_external_animal(cattle_id, category, cohort)
        self.social_model.add_external_animal(
            cattle_id,
            lot_id,
            day,
            preferred_shelter,
        )
        self.num_cattle = max(self.num_cattle, cattle_id)
        self.animal_profiles[cattle_id] = {
            "angle": self.rng.uniform(0, 2 * math.pi),
            "radius": self.rng.uniform(0.25, 1.0),
            "pace": self.rng.uniform(0.88, 1.12),
        }
        self.animal_states[cattle_id] = "normal"
        self.inventory_status[cattle_id] = "active"
        self.external_animal_records[cattle_id] = {
            "id": cattle_id,
            "category": category,
            "cohort": cohort,
            "lot_id": lot_id,
            "arrival_day": int(day),
            "preferred_shelter": preferred_shelter,
        }
        return self._add_animal_position(cattle_id, day)

    def _process_lifecycle_events(self, events, day, hour, global_frame):
        changes = []
        for event in events:
            animal_id = event.get("animal_id")
            if event["type"] == "animal_parturition" and animal_id is not None:
                event_id = event["id"]
                if event_id not in self.birth_records:
                    parameters = event.get("parameters", {})
                    delay = max(1, min(4, int(parameters.get("birth_delay_frames", 3))))
                    requested_calf_id = parameters.get("calf_id")
                    calf_id = (
                        int(requested_calf_id)
                        if requested_calf_id is not None
                        else self._next_cattle_id()
                    )
                    if calf_id in self.nutrition_model.animals:
                        calf_id = self._next_cattle_id()
                    self.birth_records[event_id] = {
                        "event_id": event_id,
                        "mother_id": int(animal_id),
                        "calf_id": calf_id,
                        "prepartum_frame": global_frame,
                        "scheduled_birth_frame": global_frame + delay,
                        "postpartum_recovery_frames": max(
                            1,
                            int(parameters.get("postpartum_recovery_frames", 12)),
                        ),
                        "postpartum_until_frame": None,
                        "mother_recovered": False,
                        "birth_day": None,
                        "birth_hour": None,
                        "birth_frame": None,
                        "born": False,
                    }
                self.animal_states[int(animal_id)] = "prepartum"
            elif event["type"] == "animal_death" and animal_id is not None:
                cattle_id = int(animal_id)
                if cattle_id not in self.death_records:
                    self.death_records[cattle_id] = {
                        "event_id": event["id"],
                        "animal_id": cattle_id,
                        "death_day": day,
                        "death_hour": hour,
                        "death_frame": global_frame,
                        "decomposition_color_change_day": max(
                            1,
                            int(
                                event.get("parameters", {}).get(
                                    "decomposition_color_change_day", 5
                                )
                            ),
                        ),
                    }
                    changes.append(
                        {"type": "animal_death", **self.death_records[cattle_id]}
                    )
                self.animal_states[cattle_id] = "dead"
                self.inventory_status[cattle_id] = "deceased"
            elif event["type"] == "animal_missing" and animal_id is not None:
                cattle_id = int(animal_id)
                if cattle_id not in self.missing_records:
                    self.missing_records[cattle_id] = {
                        "event_id": event["id"],
                        "animal_id": cattle_id,
                        "missing_day": day,
                        "missing_hour": hour,
                        "missing_frame": global_frame,
                    }

        for record in self.birth_records.values():
            mother_id = int(record["mother_id"])
            if not record["born"] and global_frame >= int(
                record["scheduled_birth_frame"]
            ):
                record["birth_day"] = day
                record["birth_hour"] = hour
                record["birth_frame"] = global_frame
                record["postpartum_until_frame"] = global_frame + int(
                    record["postpartum_recovery_frames"]
                )
                self._register_newborn(record)
                changes.append(
                    {
                        "type": "animal_birth",
                        "event_id": record["event_id"],
                        "mother_id": mother_id,
                        "calf_id": record["calf_id"],
                        "frame": global_frame,
                        "day": day,
                        "hour": hour,
                    }
                )
            record["mother_recovered"] = bool(
                record["born"]
                and global_frame > int(record.get("postpartum_until_frame") or 0)
            )
            if self.animal_states.get(mother_id) not in {"dead", "missing"}:
                self.animal_states[mother_id] = (
                    "normal"
                    if record["mother_recovered"]
                    else ("postpartum" if record["born"] else "prepartum")
                )
        return changes

    def _ground_truth_occurrences(self, day, hour):
        occurrences = {
            item["id"]: item
            for item in self.event_generator.occurrence_ground_truth(day, hour)
        }
        for record in self.death_records.values():
            occurrences.setdefault(
                record["event_id"],
                {
                    "id": record["event_id"],
                    "type": "animal_death",
                    "animal_id": int(record["animal_id"]),
                    "start_day": int(record["death_day"]),
                    "start_hour": int(record["death_hour"]),
                    "parameters": {
                        "decomposition_color_change_day": int(
                            record.get("decomposition_color_change_day", 5)
                        )
                    },
                },
            )
        for record in self.missing_records.values():
            occurrences.setdefault(
                record["event_id"],
                {
                    "id": record["event_id"],
                    "type": "animal_missing",
                    "animal_id": int(record["animal_id"]),
                    "start_day": int(record["missing_day"]),
                    "start_hour": int(record["missing_hour"]),
                    "parameters": {},
                },
            )
        return list(occurrences.values())

    def _active_ids(self, day: int):
        return [
            cattle_id
            for cattle_id in self.social_model.active_ids(day)
            if self.inventory_status.get(cattle_id) in {"active", "deceased"}
        ]

    def _apply_administrative_events(self, day: int, hour: int):
        applied = []
        for event in self.administrative_event_generator.events_at(day, hour):
            animal_ids = [int(value) for value in event.get("animal_ids", [])]
            animal_ids.extend(
                int(item["id"])
                for item in event.get("animals", [])
                if int(item["id"]) not in animal_ids
            )
            if event["type"] == "animal_sale":
                for cattle_id in animal_ids:
                    self.inventory_status[cattle_id] = "sold"
                    self.animal_states[cattle_id] = "normal"
                sold = set(animal_ids)
                self.herd_positions = [
                    animal for animal in self.herd_positions if animal["id"] not in sold
                ]
            elif event["type"] == "animal_entry":
                for specification in event.get("animals", []):
                    self._register_external_animal(specification, day)
                for cattle_id in animal_ids:
                    if self.inventory_status.get(cattle_id) != "sold":
                        self.inventory_status[cattle_id] = "active"
                        self._add_animal_position(cattle_id, day)
            elif event["type"] == "animal_transfer":
                target = event["target_paddock"]
                for cattle_id in animal_ids:
                    self.social_model.transfer(cattle_id, target)
            applied.append(
                {
                    "id": event["id"],
                    "type": event["type"],
                    "day": day,
                    "hour": hour,
                    "animal_ids": animal_ids,
                    "animals": event.get("animals", []),
                    "target_paddock": event.get("target_paddock"),
                }
            )
        return applied

    def _weather(self, hour: int, events: list[dict]):
        climate_events = [
            event
            for event in events
            if event["type"] in {"extreme_heat", "extreme_cold"}
        ]
        if climate_events:
            parameters = climate_events[-1]["parameters"]
            return int(parameters["temperature_c"]), parameters.get("wind", "Calmo")
        base_temperature = self.weather_cfg["temperature_by_hour"][str(hour)]
        variation = self.weather_cfg.get("temperature_variation_c", 0)
        temperature = base_temperature + self.rng.randint(-variation, variation)
        return temperature, self.weather_cfg["wind_by_hour"][str(hour)]

    @staticmethod
    def _progress(value: float, start: float, end: float):
        if end <= start:
            return 1.0
        return max(0.0, min(1.0, (value - start) / (end - start)))

    @staticmethod
    def _lerp(start: float, end: float, progress: float):
        return start + (end - start) * progress

    def _return_center(self, paddock: str, hour: int):
        start = self.farm.get_paddock_center(paddock)
        destination = self.farm.refuge_center(paddock)
        final_hour = max(self.sim_cfg["frame_hours"])
        progress = self._progress(
            hour,
            self.move_cfg["return_start_hour"],
            final_hour,
        )
        return (
            self._lerp(start[0], destination[0], progress),
            self._lerp(start[1], destination[1], progress),
        )

    def _sleeps_in_shelter(self, cattle_id: int, day: int):
        current = next(
            (
                position
                for position in self.herd_positions
                if position["id"] == cattle_id
            ),
            None,
        )
        if current is None or not self.farm.shelters_in(current["paddock"]):
            return False
        fraction = self.move_cfg["end_of_day"]["shelter_fraction"]
        shelter_count = min(
            round(self.num_cattle * fraction),
            self.farm.shelter_capacity(current["paddock"]),
        )
        rotating_rank = (cattle_id - 1 + day * 7) % self.num_cattle
        return rotating_rank < shelter_count

    def _refuge_behavior(self, cattle_id: int, paddock: str, reason: str, day: int):
        if self._can_use_shelter(cattle_id, paddock, day):
            return f"abrigo_{reason}"
        if self.farm.paddocks[paddock].natural_shadow > 0:
            return f"sombra_natural_{reason}"
        return "repouso_sem_abrigo"

    def _animal_decision(
        self,
        cattle_id: int,
        hour: int,
        temperature: int,
        plan,
        previous,
        events: list[dict],
    ):
        animal = self.nutrition_model.animals[cattle_id]
        category_cfg = self.nutrition_model.category_config(cattle_id)
        events_by_type = {event["type"]: event for event in events}
        cold_event = events_by_type.get("extreme_cold")
        heat_event = events_by_type.get("extreme_heat")
        cold = temperature <= self.move_cfg["cold_threshold_c"]
        release_hour = (
            self.move_cfg["cold_release_hour"]
            if cold
            else self.move_cfg["normal_release_hour"]
        )
        if cold_event:
            release_hour = cold_event["parameters"].get(
                "release_hour", release_hour
            )
        water_hour = category_cfg["water_hour"]
        afternoon_start = category_cfg["afternoon_start_hour"]
        return_hour = self.move_cfg["return_start_hour"]
        if cold_event:
            return_hour = min(
                return_hour,
                cold_event["parameters"].get("early_return_hour", return_hour),
            )
        final_hour = max(self.sim_cfg["frame_hours"])
        home_paddock = self.social_model.preferred_shelter(
            cattle_id,
            plan.home_paddock,
            plan.day,
        )
        night_paddock = self.social_model.preferred_shelter(
            cattle_id,
            plan.night_paddock,
            plan.day,
        )

        if hour < release_hour:
            behavior = (
                self._refuge_behavior(cattle_id, previous["paddock"], "frio", plan.day)
                if cold
                else self._refuge_behavior(
                    cattle_id, previous["paddock"], "rotina", plan.day
                )
            )
            return behavior, previous["paddock"]
        if hour >= final_hour:
            if self._sleeps_in_shelter(cattle_id, plan.day):
                behavior = "abrigo_fim_dia"
            elif self.farm.refuge_kind(night_paddock) == "open_field":
                behavior = "pastejo_proximo_pernoite"
            else:
                behavior = "pastejo_proximo_abrigo"
            return (behavior, night_paddock)
        if hour >= return_hour:
            return ("retorno_pastejando", night_paddock)

        if heat_event:
            parameters = heat_event["parameters"]
            refuge_start = parameters.get("refuge_start_hour", 11)
            refuge_end = parameters.get("refuge_end_hour", 15)
            if not animal.has_watered and hour >= min(water_hour, refuge_start):
                return ("busca_agua", plan.water_paddock)
            if refuge_start <= hour <= refuge_end:
                refuge_paddock = self.farm.best_refuge_paddock(previous["paddock"])
                return (
                    self._refuge_behavior(
                        cattle_id, refuge_paddock, "calor", plan.day
                    ),
                    refuge_paddock,
                )

        if not animal.has_watered and hour >= water_hour:
            return ("busca_agua", plan.water_paddock)

        if hour < water_hour:
            morning_paddock = (
                home_paddock
                if plan.morning_grazing_paddock == plan.home_paddock
                else plan.morning_grazing_paddock
            )
            if animal.target_met:
                return ("descanso_ruminacao", morning_paddock)
            return ("pastejo_manha", morning_paddock)

        if hour < afternoon_start:
            return ("descanso_ruminacao", plan.rest_paddock)

        if animal.target_met:
            return ("descanso_ruminacao", plan.rest_paddock)
        return ("pastejo_tarde", plan.afternoon_grazing_paddock)

    def _cohort_center(
        self,
        base_center,
        paddock: str,
        cattle_id: int,
        behavior: str,
        shift_multiplier: float,
    ):
        animal = self.nutrition_model.animals[cattle_id]
        stable_number = sum(ord(character) for character in animal.cohort)
        angle = math.radians(stable_number % 360)
        shift_by_behavior = {
            "pastejo_manha": 85,
            "pastejo_tarde": 90,
            "descanso_ruminacao": 45,
            "busca_agua": 30,
            "retorno_pastejando": 50,
            "pastejo_proximo_abrigo": 55,
            "pastejo_proximo_pernoite": 70,
            "abrigo": 22,
            "abrigo_frio": 22,
            "abrigo_fim_dia": 22,
            "abrigo_calor": 22,
            "sombra_natural_calor": 40,
            "sombra_natural_frio": 35,
            "repouso_sem_abrigo": 55,
            "espera_agua_indisponivel": 55,
        }
        shift = shift_by_behavior.get(behavior, 35) * shift_multiplier
        center = (
            base_center[0] + math.cos(angle) * shift,
            base_center[1] + math.sin(angle) * shift * 0.65,
        )
        return self._clamp_to_paddock(center[0], center[1], paddock)

    def _movement_target(
        self,
        cattle_id: int,
        hour: int,
        behavior: str,
        paddock: str,
        plan,
        spatial_mode: str,
        directive: dict,
        pasture_zone: str | None,
        water_source_id: str | None,
    ):
        category_cfg = self.nutrition_model.category_config(cattle_id)
        formation = self.spatial_model.parameters(spatial_mode)
        spread_factor = (
            category_cfg["cluster_spread_factor"]
            * formation["spread_multiplier"]
        )

        refuge_behavior = (
            behavior.startswith("abrigo")
            or behavior.startswith("sombra_natural")
            or behavior in {"repouso_sem_abrigo", "espera_agua_indisponivel"}
        )
        if refuge_behavior:
            cfg = self.move_cfg["shelter"]
            center = self._refuge_center_for_behavior(paddock, behavior)
            spread_x, spread_y = cfg["spread_x"], cfg["spread_y"]
            if self.farm.refuge_kind(paddock) == "open_field":
                spread_x *= 2.2
                spread_y *= 2.2
            rate = 0.18 if behavior != "abrigo_fim_dia" else 1.0
        elif behavior == "pastejo_manha":
            cfg = self.move_cfg["morning_grazing"]
            water_hour = category_cfg["water_hour"]
            progress = self._progress(hour, self.move_cfg["normal_release_hour"], water_hour - 1)
            destination = self.farm.get_paddock_center(paddock)
            start = (
                self._shelter_center(plan.home_paddock)
                if paddock == plan.home_paddock
                else destination
            )
            center = (
                self._lerp(start[0], destination[0], progress),
                self._lerp(start[1], destination[1], progress),
            )
            if pasture_zone:
                center = self.pasture_model.zone_center(pasture_zone)
            spread_x = self._lerp(cfg["start_spread"], cfg["end_spread"], progress)
            spread_y = spread_x * 0.55
            rate = self.move_cfg["default_movement_rate"]
        elif behavior == "busca_agua":
            cfg = self.move_cfg["water"]
            center = self.farm.water_sources[water_source_id].position
            spread_x, spread_y = cfg["spread_x"], cfg["spread_y"]
            rate = 1.0
        elif behavior == "descanso_ruminacao":
            cfg = self.move_cfg["rest"]
            center = cfg.get("centers", {}).get(
                paddock,
                self.farm.get_paddock_center(paddock),
            )
            spread_x, spread_y = cfg["spread_x"], cfg["spread_y"]
            rate = cfg["movement_rate"]
        elif behavior == "pastejo_tarde":
            cfg = self.move_cfg["afternoon_grazing"]
            progress = self._progress(
                hour,
                category_cfg["afternoon_start_hour"],
                self.move_cfg["return_start_hour"] - 1,
            )
            center = self.farm.get_paddock_center(paddock)
            if pasture_zone:
                center = self.pasture_model.zone_center(pasture_zone)
            spread_x = self._lerp(cfg["start_spread"], cfg["end_spread"], progress)
            spread_y = spread_x * 0.55
            rate = cfg["movement_rate"]
        elif behavior == "retorno_pastejando":
            cfg = self.move_cfg["return"]
            center = self._return_center(paddock, hour)
            spread_x, spread_y = cfg["spread_x"], cfg["spread_y"]
            rate = cfg["movement_rate"]
        elif behavior in {"pastejo_proximo_abrigo", "pastejo_proximo_pernoite"}:
            cfg = self.move_cfg["end_of_day"]
            center = self.farm.refuge_center(paddock)
            spread_x = cfg["near_shelter_spread_x"]
            spread_y = cfg["near_shelter_spread_y"]
            rate = cfg["near_shelter_movement_rate"]
        else:
            raise ValueError(f"Comportamento desconhecido: {behavior}")

        if directive["fixed_center"] is not None:
            center = directive["fixed_center"]
            spread_x, spread_y = 10, 8
        else:
            center = self._cohort_center(
                center,
                paddock,
                cattle_id,
                behavior,
                formation["cohort_shift_multiplier"],
            )
            social_x, social_y = self.social_model.social_offset(
                cattle_id,
                plan.day,
            )
            center = self._clamp_to_paddock(
                center[0] + social_x,
                center[1] + social_y,
                paddock,
            )
        return (
            center,
            max(12, spread_x * spread_factor),
            max(8, spread_y * spread_factor),
            rate,
        )

    def _profile_target(self, cattle_id, center, spread_x, spread_y, phase_shift):
        profile = self.animal_profiles[cattle_id]
        angle = profile["angle"] + phase_shift * profile["pace"]
        radius = profile["radius"]
        return (
            int(center[0] + math.cos(angle) * spread_x * radius),
            int(center[1] + math.sin(angle) * spread_y * radius),
        )

    def _isolated_target(self, cattle_id, target, previous_by_id, distance=34):
        current = previous_by_id[cattle_id]
        neighbors = [
            item
            for other_id, item in previous_by_id.items()
            if other_id != cattle_id and item["paddock"] == current["paddock"]
        ]
        if not neighbors:
            return target
        nearest = min(
            neighbors,
            key=lambda item: math.hypot(
                item["x"] - current["x"], item["y"] - current["y"]
            ),
        )
        dx = current["x"] - nearest["x"]
        dy = current["y"] - nearest["y"]
        norm = math.hypot(dx, dy) or 1.0
        return (
            target[0] + dx / norm * distance,
            target[1] + dy / norm * distance,
        )

    def _clamp_to_paddock(self, x: int, y: int, paddock_name: str):
        margin = self.move_cfg["paddock_margin_px"]
        x1, y1, x2, y2 = self.farm.get_paddock_bbox(paddock_name)
        return (
            max(x1 + margin, min(x2 - margin, int(x))),
            max(y1 + margin, min(y2 - margin, int(y))),
        )

    def _update_animal_states(self, events: list[dict]):
        fallen_ids = {
            event["animal_id"]
            for event in events
            if event["type"] == "animal_fallen"
        }
        for cattle_id, state in self.animal_states.items():
            if state == "fallen" and cattle_id not in fallen_ids:
                self.animal_states[cattle_id] = "normal"
        for event in events:
            animal_id = event.get("animal_id")
            if event["type"] == "animal_fallen" and animal_id is not None:
                self.animal_states[animal_id] = "fallen"
            elif event["type"] == "animal_missing" and animal_id is not None:
                self.animal_states[animal_id] = "missing"
            elif event["type"] == "animal_death" and animal_id is not None:
                self.animal_states[animal_id] = "dead"

        for record in self.birth_records.values():
            mother_id = int(record["mother_id"])
            if self.animal_states.get(mother_id) not in {"dead", "missing"}:
                self.animal_states[mother_id] = (
                    "normal"
                    if record.get("mother_recovered")
                    else ("postpartum" if record["born"] else "prepartum")
                )

    @staticmethod
    def _tuple_state(value):
        if isinstance(value, list):
            return tuple(SimulatorAgent._tuple_state(item) for item in value)
        return value

    def _animal_catalog(self):
        return {
            str(cattle_id): {
                "category": animal.category,
                "cohort": animal.cohort,
                "lot_id": self.social_model.lot_for(cattle_id),
                "arrival_day": self.social_model.arrival_day(cattle_id),
            }
            for cattle_id, animal in self.nutrition_model.animals.items()
        }

    def _restore_checkpoint(self, state: dict):
        self.birth_records = {
            str(key): value for key, value in state.get("birth_records", {}).items()
        }
        self.death_records = {
            int(key): value for key, value in state.get("death_records", {}).items()
        }
        self.missing_records = {
            int(key): value
            for key, value in state.get("missing_records", {}).items()
        }
        self.external_animal_records = {
            int(key): value
            for key, value in state.get("external_animal_records", {}).items()
        }
        for record in self.birth_records.values():
            if not record.get("born"):
                continue
            calf_id = int(record["calf_id"])
            mother_id = int(record["mother_id"])
            if calf_id not in self.nutrition_model.animals:
                self.nutrition_model.add_newborn(calf_id)
                self.social_model.add_newborn(calf_id, mother_id)
                self.num_cattle = max(self.num_cattle, calf_id)
                self.animal_states[calf_id] = "normal"
                self.inventory_status[calf_id] = "active"
        for record in self.external_animal_records.values():
            cattle_id = int(record["id"])
            if cattle_id in self.nutrition_model.animals:
                continue
            self.nutrition_model.add_external_animal(
                cattle_id,
                record["category"],
                record.get("cohort"),
            )
            self.social_model.add_external_animal(
                cattle_id,
                record["lot_id"],
                int(record["arrival_day"]),
                record["preferred_shelter"],
            )
            self.num_cattle = max(self.num_cattle, cattle_id)
            self.animal_states[cattle_id] = "normal"
            self.inventory_status[cattle_id] = "active"
        checkpoint_catalog = state.get("animal_catalog", {})
        current_catalog = self._animal_catalog()
        missing_ids = sorted(set(checkpoint_catalog) - set(current_catalog))
        if missing_ids:
            raise ValueError(
                "Não reduza num_bovinos ao continuar: use animal_sale. "
                f"IDs ausentes na configuração: {missing_ids}."
            )
        for cattle_id, saved in checkpoint_catalog.items():
            current = current_catalog[cattle_id]
            if current["category"] != saved["category"] or current["lot_id"] != saved["lot_id"]:
                raise ValueError(
                    f"O animal {cattle_id} mudou de categoria/lote em relação ao checkpoint."
                )
        completed_day = int(state["completed_day"])
        for cattle_id in set(current_catalog) - set(checkpoint_catalog):
            if int(current_catalog[cattle_id]["arrival_day"]) <= completed_day:
                raise ValueError(
                    f"Novo animal {cattle_id} deve ter arrival_day posterior ao dia "
                    f"já concluído ({completed_day})."
                )

        self.completed_day = completed_day
        self.global_frame = int(state["global_frame"])
        self.start_date = datetime.fromisoformat(state["start_date"])
        self.herd_positions = state["herd_positions"]
        self.inventory_status.update(
            {int(key): value for key, value in state["inventory_status"].items()}
        )
        self.animal_states.update(
            {int(key): value for key, value in state["animal_states"].items()}
        )
        self.animal_profiles.update(
            {int(key): value for key, value in state["animal_profiles"].items()}
        )
        for paddock_id, quality in state["pasture_quality"].items():
            self.farm.paddocks[paddock_id].pasture_quality = float(quality)
        for source_id, level in state["water_levels"].items():
            self.farm.water_sources[source_id].level_liters = float(level)
        for zone_id, quality in state["pasture_zones"].items():
            self.pasture_model.zones[zone_id]["quality"] = float(quality)
        self.pasture_model._update_paddock_averages()
        self.pasture_model.animal_memory = {
            int(key): value for key, value in state.get("pasture_animal_memory", {}).items()
        }
        planner = state["behavior_planner"]
        self.behavior_planner.current_home = planner["current_home"]
        self.behavior_planner.days_in_current_home = int(planner["days_in_current_home"])
        self.behavior_planner.last_scores = planner.get("last_scores", {})
        self.social_model.preferred_paddock_overrides = {
            int(key): value
            for key, value in state.get("preferred_paddock_overrides", {}).items()
        }
        self.image_generator.last_positions = {
            int(key): tuple(value)
            for key, value in state.get("image_last_positions", {}).items()
        }
        self.rng.setstate(self._tuple_state(state["rng_state"]))

    def _checkpoint_state(self):
        return {
            "start_date": self.start_date.isoformat(),
            "global_frame": self.global_frame,
            "herd_positions": self.herd_positions,
            "inventory_status": self.inventory_status,
            "animal_states": self.animal_states,
            "animal_profiles": self.animal_profiles,
            "birth_records": self.birth_records,
            "death_records": self.death_records,
            "missing_records": self.missing_records,
            "external_animal_records": self.external_animal_records,
            "animal_catalog": self._animal_catalog(),
            "pasture_quality": self.farm.pasture_snapshot(),
            "pasture_zones": self.pasture_model.zone_snapshot(),
            "pasture_animal_memory": self.pasture_model.animal_memory,
            "water_levels": {
                source_id: source.level_liters
                for source_id, source in self.farm.water_sources.items()
            },
            "behavior_planner": {
                "current_home": self.behavior_planner.current_home,
                "days_in_current_home": self.behavior_planner.days_in_current_home,
                "last_scores": self.behavior_planner.last_scores,
            },
            "preferred_paddock_overrides": (
                self.social_model.preferred_paddock_overrides
            ),
            "image_last_positions": self.image_generator.last_positions,
            "rng_state": self.rng.getstate(),
        }

    def _move_herd(self, hour: int, temperature: int, plan, events: list[dict]):
        self._update_animal_states(events)
        previous_by_id = {animal["id"]: animal for animal in self.herd_positions}
        decisions = []

        for cattle_id in self._active_ids(plan.day):
            if self.animal_states[cattle_id] == "missing":
                continue
            previous = previous_by_id.get(cattle_id)
            if previous is None:
                self._add_animal_position(cattle_id, plan.day)
                previous = next(
                    animal for animal in self.herd_positions if animal["id"] == cattle_id
                )
                previous_by_id[cattle_id] = previous
            if self.animal_states[cattle_id] in {"fallen", "dead"}:
                behavior = (
                    "animal_morto"
                    if self.animal_states[cattle_id] == "dead"
                    else "animal_caido"
                )
                paddock = previous["paddock"]
            else:
                behavior, paddock = self._animal_decision(
                    cattle_id,
                    hour,
                    temperature,
                    plan,
                    previous,
                    events,
                )
                if self.animal_states[cattle_id] in {"prepartum", "postpartum"}:
                    paddock = previous["paddock"]
            water_source_id = None
            if behavior == "busca_agua":
                candidates = [
                    source
                    for source in self.farm.available_water_sources()
                    if self.farm.is_reachable(previous["paddock"], source.paddock_id)
                ]
                if not candidates:
                    behavior = "espera_agua_indisponivel"
                    paddock = previous["paddock"]
                else:
                    preferred = next(
                        (
                            source
                            for source in candidates
                            if source.id == plan.water_source_id
                        ),
                        None,
                    )
                    selected = preferred or min(
                        candidates,
                        key=lambda source: self.farm.route_distance(
                            previous["paddock"], source.paddock_id
                        ),
                    )
                    water_source_id = selected.id
                    paddock = selected.paddock_id
            if self.animal_states[cattle_id] in {"prepartum", "postpartum"}:
                if behavior == "busca_agua":
                    behavior = "descanso_ruminacao"
                water_source_id = None
                paddock = previous["paddock"]
            nutrition = self.nutrition_model.animals[cattle_id]
            decisions.append(
                {
                    "id": cattle_id,
                    "behavior": behavior,
                    "paddock": paddock,
                    "cohort": nutrition.cohort,
                    "lot_id": self.social_model.lot_for(cattle_id),
                    "water_source_id": water_source_id,
                }
            )

        decisions_by_id = {decision["id"]: decision for decision in decisions}
        newborn_ids = {
            int(record["calf_id"]): int(record["mother_id"])
            for record in self.birth_records.values()
            if record.get("born")
        }
        for decision in decisions:
            mother_id = self.social_model.mother_of(decision["id"])
            mother_decision = decisions_by_id.get(mother_id)
            if mother_decision and decision["behavior"] in {
                "pastejo_manha",
                "pastejo_tarde",
                "descanso_ruminacao",
                "retorno_pastejando",
            }:
                decision["paddock"] = mother_decision["paddock"]
            if decision["id"] in newborn_ids and mother_decision:
                decision["behavior"] = mother_decision["behavior"]
                decision["paddock"] = mother_decision["paddock"]
                decision["water_source_id"] = mother_decision["water_source_id"]

        spatial_mode, spatial_mode_random = self.spatial_model.choose_mode(
            [decision["behavior"] for decision in decisions]
        )
        directives = self.spatial_model.prepare_directives(spatial_mode, decisions)
        new_positions = []

        for decision in decisions:
            cattle_id = decision["id"]
            previous = previous_by_id[cattle_id]
            behavior = decision["behavior"]
            planned_paddock = decision["paddock"]
            directive = directives[cattle_id]
            paddock = directive["paddock"]
            if self.animal_states[cattle_id] in {"prepartum", "postpartum"}:
                paddock = previous["paddock"]
            if behavior in {"animal_caido", "animal_morto"}:
                actual_zone = self.pasture_model.zone_at(
                    paddock, previous["x"], previous["y"]
                )
                nutrition = self.nutrition_model.animals[cattle_id]
                death_record = self.death_records.get(cattle_id)
                decomposition_days = (
                    max(0, plan.day - int(death_record["death_day"]))
                    if death_record
                    else 0
                )
                decomposition_change_day = (
                    int(death_record.get("decomposition_color_change_day", 5))
                    if death_record
                    else 5
                )
                new_positions.append(
                    {
                        **previous,
                        "state": behavior,
                        "physical_state": (
                            "dead" if behavior == "animal_morto" else "fallen"
                        ),
                        "decomposition_days": decomposition_days,
                        "decomposition_color_change_day": decomposition_change_day,
                        "decomposition_stage": (
                            "advanced"
                            if decomposition_days >= decomposition_change_day
                            else "initial"
                        ) if behavior == "animal_morto" else None,
                        "lot_id": decision["lot_id"],
                        "pasture_zone": actual_zone,
                        "pasture_quality_observed": round(
                            self.pasture_model.zone_quality(actual_zone), 4
                        ),
                        "planned_paddock": paddock,
                        "water_source_id": None,
                        "target_pasture_zone": None,
                        "pasture_zone_scores": None,
                        "spatial_mode": spatial_mode,
                        "spatial_role": "stationary_occurrence",
                        "category": nutrition.category,
                        "cohort": nutrition.cohort,
                    }
                )
                continue
            pasture_zone = None
            zone_scores = None
            if behavior in {"pastejo_manha", "pastejo_tarde"}:
                pasture_zone, zone_scores = self.pasture_model.choose_zone(
                    paddock,
                    (previous["x"], previous["y"]),
                    cattle_id,
                    decision["lot_id"],
                )
            center, spread_x, spread_y, movement_rate = self._movement_target(
                cattle_id,
                hour,
                behavior,
                paddock,
                plan,
                spatial_mode,
                directive,
                pasture_zone,
                decision["water_source_id"],
            )
            jitter = (
                self.move_cfg["rest"].get("jitter_px", 0)
                if behavior == "descanso_ruminacao"
                else self.move_cfg["grazing_jitter_px"]
            )
            target_x, target_y = self._profile_target(
                cattle_id,
                center,
                spread_x,
                spread_y,
                hour * 0.08,
            )
            physical_state = self.animal_states[cattle_id]
            if physical_state in {"prepartum", "postpartum"}:
                target_x, target_y = self._isolated_target(
                    cattle_id,
                    (target_x, target_y),
                    previous_by_id,
                    distance=38 if physical_state == "prepartum" else 24,
                )
            mother_id = self.social_model.mother_of(cattle_id)
            mother_position = previous_by_id.get(mother_id)
            if cattle_id in newborn_ids and mother_position:
                paddock = mother_position["paddock"]
                target_x = mother_position["x"] + (8 if cattle_id % 2 else -8)
                target_y = mother_position["y"] + (6 if cattle_id % 3 else -6)
                movement_rate = 0.92
            elif mother_position and mother_position["paddock"] == paddock:
                target_x, target_y = self.social_model.pull_toward_family(
                    cattle_id,
                    (target_x, target_y),
                    previous_by_id,
                )
            target_x += self.rng.randint(-jitter, jitter)
            target_y += self.rng.randint(-jitter, jitter)

            if previous["paddock"] != paddock:
                new_x, new_y = target_x, target_y
            else:
                entering_rest = (
                    behavior == "descanso_ruminacao"
                    and previous["state"] != "descanso_ruminacao"
                )
                effective_rate = 0.78 if entering_rest else movement_rate
                rate = min(
                    1.0,
                    effective_rate * self.animal_profiles[cattle_id]["pace"],
                )
                if physical_state == "prepartum":
                    rate *= 0.28
                elif physical_state == "postpartum":
                    rate *= 0.55
                new_x = previous["x"] + (target_x - previous["x"]) * rate
                new_y = previous["y"] + (target_y - previous["y"]) * rate

            if physical_state in {"prepartum", "postpartum"}:
                maximum_step = 22 if physical_state == "prepartum" else 30
                step_x = new_x - previous["x"]
                step_y = new_y - previous["y"]
                step_distance = math.hypot(step_x, step_y)
                if step_distance > maximum_step:
                    scale = maximum_step / step_distance
                    new_x = previous["x"] + step_x * scale
                    new_y = previous["y"] + step_y * scale

            new_x, new_y = self._clamp_to_paddock(new_x, new_y, paddock)
            actual_pasture_zone = self.pasture_model.zone_at(
                paddock,
                new_x,
                new_y,
            )
            nutrition = self.nutrition_model.animals[cattle_id]
            new_positions.append(
                {
                    "id": cattle_id,
                    "paddock": paddock,
                    "x": new_x,
                    "y": new_y,
                    "state": behavior,
                    "physical_state": self.animal_states[cattle_id],
                    "category": nutrition.category,
                    "cohort": nutrition.cohort,
                    "lot_id": decision["lot_id"],
                    "lot_familiarity": round(
                        self.social_model.familiarity(cattle_id, plan.day), 3
                    ),
                    "mother_id": mother_id,
                    "birth_event_id": next(
                        (
                            record["event_id"]
                            for record in self.birth_records.values()
                            if int(record.get("calf_id", -1)) == cattle_id
                        ),
                        None,
                    ),
                    "planned_paddock": planned_paddock,
                    "water_source_id": (
                        decision["water_source_id"]
                        if behavior == "busca_agua"
                        else None
                    ),
                    "pasture_zone": actual_pasture_zone,
                    "target_pasture_zone": pasture_zone,
                    "pasture_zone_scores": (
                        {name: round(score, 4) for name, score in zone_scores.items()}
                        if zone_scores
                        else None
                    ),
                    "pasture_quality_observed": (
                        round(
                            self.pasture_model.zone_quality(actual_pasture_zone), 4
                        )
                    ),
                    "spatial_mode": spatial_mode,
                    "spatial_role": directive["role"],
                }
            )

        positions_by_id = {item["id"]: item for item in new_positions}
        for calf_id, mother_id in newborn_ids.items():
            calf = positions_by_id.get(calf_id)
            mother = positions_by_id.get(mother_id)
            if calf is None or mother is None:
                continue
            offset_x = 8 if calf_id % 2 else -8
            offset_y = 6 if calf_id % 3 else -6
            calf["paddock"] = mother["paddock"]
            calf["planned_paddock"] = mother["paddock"]
            calf["x"], calf["y"] = self._clamp_to_paddock(
                mother["x"] + offset_x,
                mother["y"] + offset_y,
                mother["paddock"],
            )
            calf["state"] = "acompanha_mae"
            calf["spatial_role"] = "newborn_with_mother"

        self.herd_positions = new_positions
        return new_positions, spatial_mode, spatial_mode_random

    def _distribution(self, positions, field: str):
        return dict(Counter(position[field] for position in positions))

    def _frame_duration(self, frame_index_in_day: int):
        hours = self.sim_cfg["frame_hours"]
        if frame_index_in_day + 1 < len(hours):
            return hours[frame_index_in_day + 1] - hours[frame_index_in_day]
        return 1.0

    def run(self):
        generated = []
        first_day = self.completed_day + 1
        last_day = self.completed_day + int(self.sim_cfg["num_days"])

        for day in range(first_day, last_day + 1):
            current_date = self.start_date + timedelta(days=day - 1)
            self.pasture_model.start_day()
            self.nutrition_model.start_day()
            arrivals = self._activate_arrivals(day)
            active_ids = self._active_ids(day)
            plan = self.behavior_planner.create_plan(day)
            pasture_at_start = self.farm.pasture_snapshot()
            pasture_zones_at_start = self.pasture_model.zone_snapshot()
            self.image_generator.prepare_day(day)
            daily_spatial_modes = Counter()
            daily_random_spatial_frames = 0
            daily_administrative_events = []

            for frame_index_in_day, hour in enumerate(self.sim_cfg["frame_hours"]):
                self.global_frame += 1
                global_frame = self.global_frame
                administrative_events = self._apply_administrative_events(day, hour)
                daily_administrative_events.extend(administrative_events)
                for event in administrative_events:
                    if event["type"] == "animal_entry":
                        arrivals.extend(
                            cattle_id
                            for cattle_id in event["animal_ids"]
                            if cattle_id not in arrivals
                        )
                active_ids = self._active_ids(day)
                events = self.event_generator.active_events(day, hour)
                lifecycle_changes = self._process_lifecycle_events(
                    events,
                    day,
                    hour,
                    global_frame,
                )
                active_ids = self._active_ids(day)
                temperature, wind = self._weather(hour, events)
                positions, spatial_mode, spatial_mode_random = self._move_herd(
                    hour,
                    temperature,
                    plan,
                    events,
                )
                daily_spatial_modes[spatial_mode] += 1
                daily_random_spatial_frames += int(spatial_mode_random)
                duration = self._frame_duration(frame_index_in_day)
                self.nutrition_model.register_frame(
                    positions,
                    self.farm,
                    duration,
                )
                self.pasture_model.register_frame(positions, duration)
                drinkers_by_source = Counter(
                    position["water_source_id"]
                    for position in positions
                    if position.get("water_source_id")
                )
                heat_event = next(
                    (event for event in events if event["type"] == "extreme_heat"),
                    None,
                )
                evaporation_multiplier = (
                    heat_event["parameters"].get(
                        "water_evaporation_multiplier", 1.0
                    )
                    if heat_event
                    else 1.0
                )
                self.farm.update_water(
                    temperature,
                    duration,
                    drinkers_by_source,
                    evaporation_multiplier,
                )
                environment_state = self.farm.resource_snapshot()

                behavior_distribution = self._distribution(positions, "state")
                paddock_distribution = self._distribution(positions, "paddock")
                timestamp = current_date.replace(hour=hour, minute=0, second=0)
                metadata = {
                    "frame": global_frame,
                    "frame_in_day": frame_index_in_day + 1,
                    "day": day,
                    "timestamp": timestamp.isoformat(),
                    "hora": hour,
                    "temperatura": temperature,
                    "vento": wind,
                    "behavior": (
                        next(iter(behavior_distribution))
                        if len(behavior_distribution) == 1
                        else "comportamento_misto"
                    ),
                    "behavior_distribution": behavior_distribution,
                    "spatial_mode": spatial_mode,
                    "spatial_mode_random": spatial_mode_random,
                    "spatial_role_distribution": self._distribution(
                        positions,
                        "spatial_role",
                    ),
                    "daily_plan": plan.to_dict(),
                    "decision_scores": self.behavior_planner.last_scores,
                    "pasture_quality": pasture_at_start,
                    "pasture_zone_quality": pasture_zones_at_start,
                    "nutrition_summary": self.nutrition_model.summary(active_ids),
                    "cattle_distribution": paddock_distribution,
                    "lot_distribution": self._distribution(positions, "lot_id"),
                    "lot_state": self.social_model.snapshot(day),
                    "new_arrivals": arrivals,
                    "administrative_events": administrative_events,
                    "lifecycle_changes": lifecycle_changes,
                    "birth_records": self.birth_records,
                    "death_records": self.death_records,
                    "inventory_distribution": dict(
                        Counter(self.inventory_status.values())
                    ),
                    "environment_state": environment_state,
                    "total_cattle_expected": len(active_ids),
                    "latitude": self.config["farm"]["latitude"],
                    "longitude": self.config["farm"]["longitude"],
                    "campaign": {
                        "enabled": self.campaign_store.enabled,
                        "user_id": self.campaign_store.user_id,
                        "campaign_id": self.campaign_store.campaign_id,
                    },
                    "cattle_positions": positions,
                }

                image_path = self.image_generator.generate_frame_from_positions(
                    global_frame,
                    positions,
                    [],
                    metadata,
                )
                with open(
                    self.metadata_dir / f"frame_{global_frame:04d}.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(metadata, file, indent=2, ensure_ascii=False)
                with open(
                    self.environment_state_dir / f"frame_{global_frame:04d}.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        {
                            "frame": global_frame,
                            "timestamp": timestamp.isoformat(),
                            "temperature_c": temperature,
                            "wind": wind,
                            **environment_state,
                        },
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )
                with open(
                    self.occurrence_ground_truth_dir
                    / f"frame_{global_frame:04d}.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        {
                            "frame": global_frame,
                            "timestamp": timestamp.isoformat(),
                            "occurrences": self._ground_truth_occurrences(day, hour),
                            "lifecycle_changes": lifecycle_changes,
                            "birth_records": self.birth_records,
                            "death_records": self.death_records,
                            "missing_records": self.missing_records,
                            "animal_physical_states": {
                                str(cattle_id): state
                                for cattle_id, state in self.animal_states.items()
                                if state != "normal"
                            },
                            "visibility_ground_truth": {
                                str(cattle_id): visibility
                                for cattle_id, visibility in (
                                    self.image_generator.last_visibility_ground_truth.items()
                                )
                                if visibility["visibility_status"] != "visible"
                            },
                        },
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )
                with open(
                    self.administrative_events_dir / f"frame_{global_frame:04d}.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        {
                            "frame": global_frame,
                            "timestamp": timestamp.isoformat(),
                            "events": administrative_events,
                            "inventory_status": self.inventory_status,
                        },
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )
                generated.append(str(image_path))

            pasture_result = self.pasture_model.finish_day()
            day_summary = {
                "day": day,
                "date": current_date.date().isoformat(),
                "daily_plan": plan.to_dict(),
                "decision_scores": self.behavior_planner.last_scores,
                "nutrition_summary": self.nutrition_model.summary(active_ids),
                "lot_state": self.social_model.snapshot(day),
                "new_arrivals": arrivals,
                "administrative_events": daily_administrative_events,
                "inventory_distribution": dict(Counter(self.inventory_status.values())),
                "spatial_mode_distribution": dict(daily_spatial_modes),
                "random_spatial_frames": daily_random_spatial_frames,
                **pasture_result,
            }
            with open(
                self.day_summaries_dir / f"day_{day:02d}.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(day_summary, file, indent=2, ensure_ascii=False)

            self.completed_day = day
            checkpoint_interval = max(
                1,
                int(
                    self.config.get("campaign", {}).get(
                        "checkpoint_every_days", 1
                    )
                ),
            )
            if day % checkpoint_interval == 0 or day == last_day:
                self.campaign_store.save_checkpoint(day, self._checkpoint_state())

        self.campaign_store.finish_run(self.completed_day, self.global_frame)
        return generated
