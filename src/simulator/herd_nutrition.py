from dataclasses import dataclass


@dataclass
class AnimalNutrition:
    cattle_id: int
    category: str
    cohort: str
    daily_intake_target_grams: float
    intake_grams: float = 0.0
    has_watered: bool = False

    @property
    def target_met(self):
        return self.intake_grams >= self.daily_intake_target_grams


class HerdNutritionModel:
    """Mantém consumo, sede e necessidades distintas por categoria animal."""

    def __init__(self, config: dict, num_cattle: int):
        self.config = config
        self.animals = self._create_animals(num_cattle)

    def _create_animals(self, num_cattle: int):
        animals = {}
        cattle_id = 1

        for category, category_cfg in self.config["categories"].items():
            count = category_cfg["count"]
            cohort_count = category_cfg["cohort_count"]
            for index in range(count):
                if cattle_id > num_cattle:
                    break
                cohort_number = index % cohort_count + 1
                animals[cattle_id] = AnimalNutrition(
                    cattle_id=cattle_id,
                    category=category,
                    cohort=f"{category}_{cohort_number}",
                    daily_intake_target_grams=category_cfg[
                        "daily_intake_target_grams"
                    ],
                )
                cattle_id += 1

        if len(animals) != num_cattle:
            configured = sum(
                category["count"] for category in self.config["categories"].values()
            )
            raise ValueError(
                f"A soma das categorias ({configured}) deve ser igual a "
                f"num_bovinos ({num_cattle})."
            )
        return animals

    def start_day(self):
        for animal in self.animals.values():
            animal.intake_grams = 0.0
            animal.has_watered = False

    def category_config(self, cattle_id: int):
        animal = self.animals[cattle_id]
        return self.config["categories"][animal.category]

    def add_newborn(self, cattle_id: int):
        """Registra um terneiro nascido durante uma simulação em andamento."""

        cattle_id = int(cattle_id)
        if cattle_id in self.animals:
            return self.animals[cattle_id]
        category = "terneiro"
        category_cfg = self.config["categories"][category]
        animal = AnimalNutrition(
            cattle_id=cattle_id,
            category=category,
            cohort="terneiros_nascidos",
            daily_intake_target_grams=category_cfg["daily_intake_target_grams"],
        )
        self.animals[cattle_id] = animal
        return animal

    def add_external_animal(self, cattle_id: int, category: str, cohort=None):
        """Registra um animal incorporado ao inventário durante uma campanha."""

        cattle_id = int(cattle_id)
        if cattle_id in self.animals:
            return self.animals[cattle_id]
        if category not in self.config["categories"]:
            raise ValueError(f"Categoria nutricional desconhecida: {category}.")
        category_cfg = self.config["categories"][category]
        animal = AnimalNutrition(
            cattle_id=cattle_id,
            category=category,
            cohort=cohort or f"{category}_externo",
            daily_intake_target_grams=category_cfg["daily_intake_target_grams"],
        )
        self.animals[cattle_id] = animal
        return animal

    def register_frame(self, positions: list[dict], environment, duration_hours: float):
        for position in positions:
            animal = self.animals[position["id"]]
            frame_intake = 0.0
            if position["state"] in {
                "pastejo_manha",
                "pastejo_tarde",
                "pastejo_proximo_abrigo",
                "pastejo_proximo_pernoite",
            }:
                quality = position.get("pasture_quality_observed")
                if quality is None:
                    quality = environment.paddocks[position["paddock"]].pasture_quality
                efficiency = (
                    self.config["minimum_pasture_efficiency"]
                    + quality * self.config["pasture_quality_efficiency"]
                )
                intake = (
                    self.config["base_bites_per_hour"]
                    * self.config["bite_grams"]
                    * efficiency
                    * duration_hours
                )
                previous_intake = animal.intake_grams
                animal.intake_grams = min(
                    animal.daily_intake_target_grams,
                    animal.intake_grams + intake,
                )
                frame_intake = animal.intake_grams - previous_intake
            elif position["state"] == "busca_agua":
                animal.has_watered = True

            position["category"] = animal.category
            position["cohort"] = animal.cohort
            position["intake_grams"] = round(animal.intake_grams, 1)
            position["frame_intake_grams"] = round(frame_intake, 1)
            position["intake_target_grams"] = animal.daily_intake_target_grams
            position["intake_target_met"] = animal.target_met

    def summary(self, active_ids=None):
        active_ids = set(active_ids or self.animals)
        result = {}
        for category in self.config["categories"]:
            animals = [
                animal
                for animal in self.animals.values()
                if animal.category == category and animal.cattle_id in active_ids
            ]
            if not animals:
                continue
            result[category] = {
                "animals": len(animals),
                "average_intake_grams": round(
                    sum(animal.intake_grams for animal in animals) / len(animals),
                    1,
                ),
                "average_target_grams": round(
                    sum(animal.daily_intake_target_grams for animal in animals)
                    / len(animals),
                    1,
                ),
                "target_met_count": sum(animal.target_met for animal in animals),
                "watered_count": sum(animal.has_watered for animal in animals),
            }
        return result
