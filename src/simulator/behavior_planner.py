from dataclasses import asdict, dataclass
import math


@dataclass
class DailyPlan:
    day: int
    home_paddock: str
    morning_grazing_paddock: str
    water_paddock: str
    water_source_id: str
    rest_paddock: str
    afternoon_grazing_paddock: str
    night_paddock: str

    def to_dict(self):
        return asdict(self)


class BehaviorPlanner:
    """Escolhe apenas destinos existentes, acessíveis e com recursos disponíveis."""

    def __init__(self, environment, pasture_config: dict, planner_config: dict):
        self.environment = environment
        self.pasture_config = pasture_config
        self.config = planner_config
        self.current_home = planner_config["initial_home_paddock"]
        self.days_in_current_home = 0
        self.last_scores = {}

    def _distance(self, origin: str, destination: str):
        diagonal = math.hypot(self.environment.width, self.environment.height)
        return self.environment.route_distance(origin, destination) / diagonal

    def _accessible(self, origin=None):
        return [
            paddock
            for paddock in self.environment.paddocks.values()
            if paddock.accessible
            and (origin is None or self.environment.is_reachable(origin, paddock.name))
        ]

    def _choose_night_paddock(self, home_paddock: str):
        candidates = [
            paddock
            for paddock in self._accessible(home_paddock)
            if paddock.has_shadow
        ]
        if not candidates:
            return home_paddock, {home_paddock: 0.0}

        scores = {}
        for paddock in candidates:
            score = paddock.pasture_quality * self.config["shelter_pasture_weight"]
            if paddock.name == home_paddock:
                score += self.config["home_inertia_bonus"]
            scores[paddock.name] = score

        current_quality = self.environment.paddocks[home_paddock].pasture_quality
        best_name = max(scores, key=scores.get)
        can_switch = self.days_in_current_home >= self.config["minimum_home_stay_days"]
        critical = current_quality <= self.pasture_config["critical_quality"]
        current_score = scores.get(home_paddock, current_quality)
        clearly_better = scores[best_name] > (
            current_score + self.config["switch_score_margin"]
        )
        if best_name != home_paddock and (critical or (can_switch and clearly_better)):
            return best_name, scores
        return home_paddock, scores

    def _grazing_scores(self, origin: str):
        scores = {}
        for paddock in self._accessible(origin):
            score = (
                paddock.pasture_quality * self.config["pasture_weight"]
                - self._distance(origin, paddock.name) * self.config["distance_weight"]
            )
            if paddock.name == origin:
                score += self.config["same_paddock_bonus"]
            scores[paddock.name] = score
        return scores

    def create_plan(self, day: int):
        home = self.current_home
        night, home_scores = self._choose_night_paddock(home)
        home_quality = self.environment.paddocks[home].pasture_quality
        morning_scores = self._grazing_scores(home)
        morning = (
            home
            if home_quality >= self.pasture_config["preferred_quality"]
            else max(morning_scores, key=morning_scores.get)
        )

        water_candidates = [
            source
            for source in self.environment.available_water_sources()
            if self.environment.is_reachable(morning, source.paddock_id)
        ]
        if not water_candidates:
            raise ValueError("Nenhuma fonte de água acessível está disponível.")
        water_source = min(
            water_candidates,
            key=lambda source: (
                self._distance(morning, source.paddock_id)
                - 0.08 * (source.level_liters / source.capacity_liters)
            ),
        )
        water = water_source.paddock_id
        rest = water
        afternoon_scores = self._grazing_scores(rest)
        afternoon = max(afternoon_scores, key=afternoon_scores.get)

        plan = DailyPlan(
            day=day,
            home_paddock=home,
            morning_grazing_paddock=morning,
            water_paddock=water,
            water_source_id=water_source.id,
            rest_paddock=rest,
            afternoon_grazing_paddock=afternoon,
            night_paddock=night,
        )
        self.last_scores = {
            "home": {name: round(value, 4) for name, value in home_scores.items()},
            "morning_grazing": {
                name: round(value, 4) for name, value in morning_scores.items()
            },
            "afternoon_grazing": {
                name: round(value, 4) for name, value in afternoon_scores.items()
            },
        }
        if night != home:
            self.current_home = night
            self.days_in_current_home = 0
        else:
            self.days_in_current_home += 1
        return plan
