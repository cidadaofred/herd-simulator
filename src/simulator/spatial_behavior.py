class SpatialBehaviorModel:
    """Combina formações estruturadas com variações espaciais pseudoaleatórias."""

    def __init__(self, environment, config: dict, rng):
        self.environment = environment
        self.config = config
        self.rng = rng
        self.eligible_behaviors = set(config["eligible_behaviors"])
        self.current_mode = None
        self.remaining_frames = 0

    def _weighted_random_mode(self):
        modes = list(self.config["random_mode_weights"])
        weights = [self.config["random_mode_weights"][mode] for mode in modes]
        return self.rng.choices(modes, weights=weights, k=1)[0]

    def choose_mode(self, behaviors: list[str]):
        behavior_set = set(behaviors)
        if behavior_set == {"retorno_pastejando"}:
            self.current_mode = "retorno_disperso"
            self.remaining_frames = 0
            return self.current_mode, False
        if behavior_set and behavior_set <= {
            "abrigo_fim_dia",
            "pastejo_proximo_abrigo",
        }:
            self.current_mode = "fim_dia_misto"
            self.remaining_frames = 0
            return self.current_mode, False
        if behavior_set == {"pastejo_proximo_pernoite"}:
            self.current_mode = "fim_dia_campo_aberto"
            self.remaining_frames = 0
            return self.current_mode, False

        eligible = any(
            behavior in self.eligible_behaviors for behavior in behaviors
        )
        if not eligible:
            self.current_mode = "contextual_compacto"
            self.remaining_frames = 0
            return self.current_mode, False

        if self.remaining_frames <= 0 or self.current_mode == "contextual_compacto":
            if self.rng.random() < self.config["random_mode_probability"]:
                self.current_mode = self._weighted_random_mode()
            else:
                self.current_mode = "estruturado"
            duration = self.rng.randint(
                self.config["persistence_frames_min"],
                self.config["persistence_frames_max"],
            )
            self.remaining_frames = duration

        self.remaining_frames -= 1
        return self.current_mode, self.current_mode != "estruturado"

    def parameters(self, mode: str):
        if mode == "contextual_compacto":
            return {
                "spread_multiplier": 0.72,
                "cohort_shift_multiplier": 0.55,
            }
        return self.config["formation_parameters"][mode]

    def _alternative_paddock(self, current: str):
        candidates = [
            paddock
            for paddock in self.environment.paddocks.values()
            if paddock.accessible and paddock.name != current
        ]
        if not candidates:
            return current
        weights = [0.10 + paddock.pasture_quality for paddock in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0].name

    def _random_point(self, paddock_name: str):
        margin = self.config["isolated_margin_px"]
        x1, y1, x2, y2 = self.environment.get_paddock_bbox(paddock_name)
        return (
            self.rng.randint(x1 + margin, x2 - margin),
            self.rng.randint(y1 + margin, y2 - margin),
        )

    def prepare_directives(self, mode: str, decisions: list[dict]):
        directives = {
            decision["id"]: {
                "paddock": decision["paddock"],
                "role": "formacao_base",
                "fixed_center": None,
            }
            for decision in decisions
        }
        grazing = [
            decision
            for decision in decisions
            if decision["behavior"] in {"pastejo_manha", "pastejo_tarde"}
        ]
        formation_eligible = [
            decision
            for decision in decisions
            if decision["behavior"] in self.eligible_behaviors
        ]

        if mode == "subgrupos_exploradores" and grazing:
            cohorts = sorted({decision["cohort"] for decision in grazing})
            count = min(
                len(cohorts),
                self.rng.randint(
                    self.config["exploring_cohorts_min"],
                    self.config["exploring_cohorts_max"],
                ),
            )
            selected = set(self.rng.sample(cohorts, count))
            destination_by_cohort = {}
            for decision in grazing:
                if decision["cohort"] not in selected:
                    continue
                destination = destination_by_cohort.setdefault(
                    decision["cohort"],
                    self._alternative_paddock(decision["paddock"]),
                )
                directives[decision["id"]].update(
                    paddock=destination,
                    role="subgrupo_explorador",
                )

        if mode == "individuos_desconexos" and grazing:
            fraction = self.rng.uniform(
                self.config["isolated_fraction_min"],
                self.config["isolated_fraction_max"],
            )
            count = max(1, round(len(grazing) * fraction))
            selected_ids = {
                decision["id"] for decision in self.rng.sample(grazing, count)
            }
            for decision in grazing:
                if decision["id"] not in selected_ids:
                    continue
                directives[decision["id"]].update(
                    role="individuo_desconexo",
                    fixed_center=self._random_point(decision["paddock"]),
                )

        if mode == "subgrupos_distantes":
            for decision in formation_eligible:
                directives[decision["id"]]["role"] = "subgrupo_distante"
        elif mode == "grupo_disperso":
            for decision in formation_eligible:
                directives[decision["id"]]["role"] = "grupo_disperso"
        elif mode == "grupo_compacto":
            for decision in formation_eligible:
                directives[decision["id"]]["role"] = "grupo_compacto"

        return directives
