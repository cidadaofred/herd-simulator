import math


class PastureModel:
    """Atualiza pastos por zonas e orienta a busca por áreas mais produtivas."""

    ZONE_LAYOUT = {
        "NW": (0.25, 0.25),
        "NE": (0.75, 0.25),
        "SW": (0.25, 0.75),
        "SE": (0.75, 0.75),
    }

    def __init__(self, environment, config: dict):
        self.environment = environment
        self.config = config
        self.zone_cfg = config["zone_model"]
        self.zones = {}
        self.animal_memory = {}

        self._create_zones()
        self.start_day()

    def _create_zones(self):
        offsets = self.zone_cfg["initial_quality_offsets"]
        for paddock_name, paddock in self.environment.paddocks.items():
            x1, y1, x2, y2 = paddock.bbox
            for zone_name, (x_ratio, y_ratio) in self.ZONE_LAYOUT.items():
                full_name = f"{paddock_name}_{zone_name}"
                self.zones[full_name] = {
                    "name": full_name,
                    "paddock": paddock_name,
                    "center": (
                        x1 + (x2 - x1) * x_ratio,
                        y1 + (y2 - y1) * y_ratio,
                    ),
                    "quality": max(
                        0.0,
                        min(1.0, paddock.pasture_quality + offsets[zone_name]),
                    ),
                }
        self._update_paddock_averages()

    def _update_paddock_averages(self):
        for paddock_name, paddock in self.environment.paddocks.items():
            qualities = [
                zone["quality"]
                for zone in self.zones.values()
                if zone["paddock"] == paddock_name
            ]
            paddock.pasture_quality = sum(qualities) / len(qualities)

    def start_day(self):
        self.daily_usage = {name: 0.0 for name in self.zones}
        self.daily_consumed_grams = {name: 0.0 for name in self.zones}

    def zone_center(self, zone_name: str):
        return self.zones[zone_name]["center"]

    def zone_quality(self, zone_name: str):
        return self.zones[zone_name]["quality"]

    def zones_for_paddock(self, paddock_name: str):
        return [
            zone for zone in self.zones.values() if zone["paddock"] == paddock_name
        ]

    def choose_zone(self, paddock_name: str, origin, cattle_id: int, lot_id: str):
        utility = self.zone_cfg["utility_weights"]
        diagonal = math.hypot(self.environment.width, self.environment.height)
        remembered = self.animal_memory.get(cattle_id)
        scores = {}
        for zone in self.zones_for_paddock(paddock_name):
            distance = math.hypot(
                zone["center"][0] - origin[0],
                zone["center"][1] - origin[1],
            ) / diagonal
            crowding = self.daily_usage[zone["name"]] / max(
                1.0,
                self.environment.paddocks[paddock_name].carrying_capacity
                / len(self.ZONE_LAYOUT),
            )
            memory_bonus = (
                self.zone_cfg["memory_bonus"]
                if remembered == zone["name"]
                else 0.0
            )
            stable_bias = (
                sum(ord(character) for character in f"{lot_id}:{zone['name']}") % 13
            ) / 1000
            scores[zone["name"]] = (
                zone["quality"] * utility["quality"]
                - distance * utility["distance"]
                - crowding * utility["crowding"]
                + memory_bonus
                + stable_bias
            )
        return max(scores, key=scores.get), scores

    def zone_at(self, paddock_name: str, x: float, y: float):
        center_x, center_y = self.environment.get_paddock_center(paddock_name)
        horizontal = "W" if x < center_x else "E"
        vertical = "N" if y < center_y else "S"
        return f"{paddock_name}_{vertical}{horizontal}"

    def register_frame(self, positions: list[dict], duration_hours: float):
        weights = self.config["behavior_usage_weights"]
        for animal in positions:
            zone_name = animal.get("pasture_zone") or self.zone_at(
                animal["paddock"], animal["x"], animal["y"]
            )
            animal["pasture_zone"] = zone_name
            animal["pasture_quality_observed"] = round(
                self.zone_quality(zone_name), 4
            )
            behavior = animal["state"]
            self.daily_usage[zone_name] += (
                duration_hours * weights.get(behavior, 0.0)
            )
            self.daily_consumed_grams[zone_name] += animal.get(
                "frame_intake_grams", 0.0
            )
            if weights.get(behavior, 0.0) >= 0.5:
                self.animal_memory[animal["id"]] = zone_name

    def finish_day(self):
        before = self.environment.pasture_snapshot()
        zones_before = self.zone_snapshot()
        zone_details = {}

        for zone_name, zone in self.zones.items():
            paddock = zone["paddock"]
            zone_capacity = self.environment.paddocks[paddock].carrying_capacity / len(
                self.ZONE_LAYOUT
            )
            reference_hours = self.config["reference_grazing_hours"]
            trampling_load = max(1.0, zone_capacity * reference_hours)
            trampling_pressure = self.daily_usage[zone_name] / trampling_load
            reference_intake = (
                zone_capacity * self.config["reference_intake_grams_per_animal"]
            )
            intake_pressure = self.daily_consumed_grams[zone_name] / reference_intake
            pressure = max(intake_pressure, trampling_pressure)
            recovery = (
                self.config["daily_recovery_rate"]
                * (1.0 - zone["quality"])
                * max(0.0, 1.0 - min(1.0, pressure))
            )
            consumption = self.config["daily_consumption_rate"] * intake_pressure
            trampling = self.config["trampling_rate"] * min(
                1.5, trampling_pressure
            )
            zone["quality"] = max(
                0.0,
                min(1.0, zone["quality"] + recovery - consumption - trampling),
            )
            zone_details[zone_name] = {
                "consumed_grams": round(self.daily_consumed_grams[zone_name], 1),
                "animal_equivalent_hours": round(self.daily_usage[zone_name], 2),
                "intake_pressure": round(intake_pressure, 4),
                "trampling_pressure": round(trampling_pressure, 4),
                "quality_after": round(zone["quality"], 4),
            }

        self._update_paddock_averages()
        paddock_details = {}
        for paddock_name in self.environment.paddocks:
            zone_names = [
                name for name, zone in self.zones.items() if zone["paddock"] == paddock_name
            ]
            paddock_details[paddock_name] = {
                "consumed_grams": round(
                    sum(self.daily_consumed_grams[name] for name in zone_names), 1
                ),
                "animal_equivalent_hours": round(
                    sum(self.daily_usage[name] for name in zone_names), 2
                ),
            }

        return {
            "pasture_before": before,
            "pasture_after": self.environment.pasture_snapshot(),
            "pasture_dynamics": paddock_details,
            "pasture_zones_before": zones_before,
            "pasture_zones_after": self.zone_snapshot(),
            "pasture_zone_dynamics": zone_details,
        }

    def zone_snapshot(self):
        return {
            name: round(zone["quality"], 4) for name, zone in self.zones.items()
        }
