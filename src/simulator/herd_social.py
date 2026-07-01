import math


class HerdSocialModel:
    """Mantém lotes, adaptação social e vínculos entre mães e terneiros."""

    def __init__(self, config: dict, num_cattle: int):
        self.config = config
        self.num_cattle = num_cattle
        self.lots = {}
        self.animal_lot = {}
        self.mother_by_calf = {}
        self.preferred_paddock_overrides = {}
        self._load_lots()

    @staticmethod
    def _expand_ids(value):
        if isinstance(value, list):
            return [int(item) for item in value]
        start, end = (int(item) for item in str(value).split("-", 1))
        return list(range(start, end + 1))

    def _load_lots(self):
        for lot_cfg in self.config["lots"]:
            if not lot_cfg.get("enabled", True):
                continue
            lot_id = lot_cfg["id"]
            animal_ids = self._expand_ids(lot_cfg["animal_ids"])
            self.lots[lot_id] = {**lot_cfg, "animal_ids": animal_ids}
            for cattle_id in animal_ids:
                if cattle_id in self.animal_lot:
                    raise ValueError(f"O animal {cattle_id} aparece em mais de um lote.")
                self.animal_lot[cattle_id] = lot_id

        missing = set(range(1, self.num_cattle + 1)) - set(self.animal_lot)
        if missing:
            raise ValueError(f"Animais sem lote configurado: {sorted(missing)}")

    def configure_family_links(self, animals: dict):
        calves = [a.cattle_id for a in animals.values() if a.category == "terneiro"]
        mothers = [
            a.cattle_id for a in animals.values() if a.category == "vaca_lactante"
        ]
        if not mothers:
            return
        for index, calf_id in enumerate(calves):
            self.mother_by_calf[calf_id] = mothers[index % len(mothers)]

    def add_newborn(self, calf_id: int, mother_id: int):
        """Inclui o recém-nascido no lote da mãe e cria o vínculo familiar."""

        calf_id = int(calf_id)
        mother_id = int(mother_id)
        if calf_id in self.animal_lot:
            return
        lot_id = self.lot_for(mother_id)
        self.animal_lot[calf_id] = lot_id
        self.lots[lot_id]["animal_ids"].append(calf_id)
        self.mother_by_calf[calf_id] = mother_id
        self.num_cattle = max(self.num_cattle, calf_id)

    def add_external_animal(
        self,
        cattle_id: int,
        lot_id: str,
        arrival_day: int,
        preferred_shelter: str,
    ):
        cattle_id = int(cattle_id)
        if cattle_id in self.animal_lot:
            return
        if lot_id not in self.lots:
            self.lots[lot_id] = {
                "id": lot_id,
                "animal_ids": [],
                "arrival_day": int(arrival_day),
                "preferred_shelter": preferred_shelter,
                "enabled": True,
            }
        self.lots[lot_id]["animal_ids"].append(cattle_id)
        self.animal_lot[cattle_id] = lot_id
        self.num_cattle = max(self.num_cattle, cattle_id)

    def lot_for(self, cattle_id: int):
        return self.animal_lot[cattle_id]

    def lot_config(self, cattle_id: int):
        return self.lots[self.lot_for(cattle_id)]

    def arrival_day(self, cattle_id: int):
        return int(self.lot_config(cattle_id)["arrival_day"])

    def active_ids(self, day: int):
        return [
            cattle_id
            for cattle_id in range(1, self.num_cattle + 1)
            if self.arrival_day(cattle_id) <= day
        ]

    def arrivals(self, day: int):
        return [
            cattle_id
            for cattle_id in range(1, self.num_cattle + 1)
            if self.arrival_day(cattle_id) == day
        ]

    def familiarity(self, cattle_id: int, day: int):
        age = max(0, day - self.arrival_day(cattle_id))
        integration_days = max(1, int(self.config["integration_days"]))
        return min(1.0, age / integration_days)

    def preferred_shelter(self, cattle_id: int, planned: str, day: int):
        if cattle_id in self.preferred_paddock_overrides:
            return self.preferred_paddock_overrides[cattle_id]
        familiarity = self.familiarity(cattle_id, day)
        threshold = self.config["separate_shelter_until_familiarity"]
        if familiarity < threshold:
            return self.lot_config(cattle_id)["preferred_shelter"]
        return planned

    def transfer(self, cattle_id: int, target_paddock: str):
        self.preferred_paddock_overrides[int(cattle_id)] = target_paddock

    def mother_of(self, cattle_id: int):
        return self.mother_by_calf.get(cattle_id)

    def social_offset(self, cattle_id: int, day: int):
        active_lots = [
            lot
            for lot in self.lots.values()
            if int(lot["arrival_day"]) <= day
        ]
        if len(active_lots) <= 1:
            return 0.0, 0.0
        lot_id = self.lot_for(cattle_id)
        stable_number = sum(ord(character) for character in lot_id)
        angle = math.radians(stable_number % 360)
        familiarity = self.familiarity(cattle_id, day)
        new_distance = self.config["new_lot_separation_px"]
        established_distance = self.config["established_lot_separation_px"]
        distance = new_distance * (1.0 - familiarity) + established_distance * familiarity
        return math.cos(angle) * distance, math.sin(angle) * distance * 0.65

    def pull_toward_family(self, cattle_id: int, target, positions_by_id: dict):
        mother_id = self.mother_of(cattle_id)
        mother = positions_by_id.get(mother_id)
        if mother is None:
            return target
        attraction = self.config["mother_attraction"]
        return (
            target[0] * (1.0 - attraction) + mother["x"] * attraction,
            target[1] * (1.0 - attraction) + mother["y"] * attraction,
        )

    def snapshot(self, day: int):
        result = {}
        for lot_id, lot in self.lots.items():
            active = [animal for animal in lot["animal_ids"] if self.arrival_day(animal) <= day]
            if not active:
                continue
            result[lot_id] = {
                "arrival_day": lot["arrival_day"],
                "active_animals": len(active),
                "familiarity": round(self.familiarity(active[0], day), 3),
                "preferred_shelter": lot["preferred_shelter"],
            }
        return result
