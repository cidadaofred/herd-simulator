import math
from collections import defaultdict, deque


class RuleEngine:
    """Infere ocorrências somente a partir de observações temporais."""

    def __init__(self, config):
        self.config = config
        self.stationary_cfg = config["stationary"]
        self.missing_cfg = config["missing"]
        self.death_cfg = config.get("death", {})
        self.parturition_cfg = config.get("parturition", {})
        self.population_cfg = config.get("population_adaptation", {})
        window = max(
            int(self.stationary_cfg["window_frames"]),
            int(self.parturition_cfg.get("prepartum_window_frames", 4)),
        )
        self.histories = defaultdict(lambda: deque(maxlen=window))
        death_history_size = max(
            16,
            int(self.death_cfg.get("history_frames", 72)),
        )
        self.death_histories = defaultdict(
            lambda: deque(maxlen=death_history_size)
        )
        self.known_tracks = set()
        self.absence_frames = defaultdict(int)
        self.processed_frames = 0
        self.birth_candidates = {}
        self.confirmed_parturitions = set()
        self.confirmed_deaths = set()
        self.reference_tracks = set()
        self.population_presence = defaultdict(int)
        self.population_absence = defaultdict(int)
        self.population_last_seen = {}
        self.population_history = []
        self.population_initialized = False

    def _update_population_reference(self, observation, current):
        """Atualiza a população de referência usando somente persistência visual."""
        minimum_confidence = float(
            self.population_cfg.get("minimum_detection_confidence", 0.55)
        )
        confirmation_frames = int(
            self.population_cfg.get("new_track_confirmation_frames", 2)
        )
        retirement_frames = int(
            self.population_cfg.get("retirement_observed_absence_frames", 6)
        )
        start_hour = int(
            self.population_cfg.get(
                "adaptation_start_hour", self.missing_cfg["analysis_start_hour"]
            )
        )
        end_hour = int(
            self.population_cfg.get(
                "adaptation_end_hour", self.missing_cfg["analysis_end_hour"]
            )
        )
        frame = int(observation["frame"])
        reliable_tracks = {
            track_id
            for track_id, detection in current.items()
            if float(detection.get("confidence", 0.0)) >= minimum_confidence
        }

        if not self.population_initialized and not reliable_tracks:
            observation["population_estimate"] = {
                "count": 0,
                "reference_source": "awaiting_persistent_visual_tracks",
                "cause_known": False,
            }
            return

        if not self.population_initialized:
            self.reference_tracks.update(reliable_tracks)
            self.population_initialized = True
            self.population_history.append(
                {
                    "frame": frame,
                    "timestamp": observation["timestamp"],
                    "change": "initial_visual_reference",
                    "added_track_ids": sorted(reliable_tracks),
                    "retired_track_ids": [],
                    "estimated_population": len(self.reference_tracks),
                    "cause": "unknown_not_inferred",
                }
            )

        added = []
        for track_id in reliable_tracks:
            if track_id in self.reference_tracks:
                self.population_presence[track_id] = 0
                self.population_absence[track_id] = 0
                self.population_last_seen[track_id] = frame
                continue
            last_seen = self.population_last_seen.get(track_id)
            self.population_presence[track_id] = (
                self.population_presence[track_id] + 1
                if last_seen == frame - 1
                else 1
            )
            self.population_last_seen[track_id] = frame
            if self.population_presence[track_id] >= confirmation_frames:
                self.reference_tracks.add(track_id)
                self.population_presence[track_id] = 0
                self.population_absence[track_id] = 0
                added.append(track_id)

        retired = []
        eligible_frame = start_hour <= int(observation["hour"]) <= end_hour
        if eligible_frame and reliable_tracks:
            for track_id in list(self.reference_tracks):
                if track_id in reliable_tracks:
                    self.population_absence[track_id] = 0
                    continue
                self.population_absence[track_id] += 1
                if self.population_absence[track_id] >= retirement_frames:
                    self.reference_tracks.remove(track_id)
                    retired.append(track_id)

        if added or retired:
            self.population_history.append(
                {
                    "frame": frame,
                    "timestamp": observation["timestamp"],
                    "change": "persistent_visual_population_change",
                    "added_track_ids": sorted(added),
                    "retired_track_ids": sorted(retired),
                    "estimated_population": len(self.reference_tracks),
                    "cause": "unknown_not_inferred",
                }
            )
        observation["population_estimate"] = {
            "count": len(self.reference_tracks),
            "reference_source": "persistent_visual_tracks",
            "cause_known": False,
        }

    @staticmethod
    def _distance(first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _stationary_predictions(self, observation, current):
        predictions = []
        window = int(self.stationary_cfg["window_frames"])
        maximum_span = float(self.stationary_cfg["maximum_span_px"])
        minimum_isolation = float(self.stationary_cfg["minimum_isolation_px"])
        minimum_confidence = float(self.stationary_cfg["minimum_confidence"])

        for track_id, detection in current.items():
            history = self.histories[track_id]
            history.append(
                {
                    "frame": observation["frame"],
                    "center": detection["center_xy"],
                    "confidence": detection["confidence"],
                }
            )
            if len(history) < window:
                continue
            frames = [item["frame"] for item in history]
            if frames[-1] - frames[0] != window - 1:
                continue
            centers = [item["center"] for item in history]
            span = max(
                self._distance(first, second)
                for first in centers
                for second in centers
            )
            if span > maximum_span:
                continue
            neighbors = [
                self._distance(detection["center_xy"], other["center_xy"])
                for other_id, other in current.items()
                if other_id != track_id
            ]
            nearest = min(neighbors) if neighbors else float("inf")
            if nearest < minimum_isolation:
                continue
            detector_confidence = sum(
                item["confidence"] for item in history
            ) / len(history)
            movement_score = max(0.0, 1.0 - span / maximum_span)
            isolation_score = min(1.0, nearest / (minimum_isolation * 2))
            confidence = (
                detector_confidence * 0.45
                + movement_score * 0.35
                + isolation_score * 0.20
            )
            if confidence < minimum_confidence:
                continue
            predictions.append(
                {
                    "type": "animal_fallen",
                    "track_id": track_id,
                    "frame": observation["frame"],
                    "timestamp": observation["timestamp"],
                    "confidence": round(min(0.99, confidence), 3),
                    "evidence": {
                        "stationary_window_frames": window,
                        "position_span_px": round(span, 2),
                        "nearest_neighbor_px": (
                            None if math.isinf(nearest) else round(nearest, 2)
                        ),
                        "mean_detection_confidence": round(detector_confidence, 3),
                    },
                }
            )
        return predictions

    def _missing_predictions(self, observation, current):
        predictions = []
        # A referencia vem do proprio historico visual, nunca do inventario
        # real mantido pelo simulador.
        reference_tracks = set(self.reference_tracks)
        expected = max(1, len(reference_tracks))
        observed_count = len(current)
        observed_ratio = observed_count / expected
        eligible_hour = (
            int(self.missing_cfg["analysis_start_hour"])
            <= int(observation["hour"])
            <= int(self.missing_cfg["analysis_end_hour"])
        )
        reliable_frame = (
            observed_ratio
            >= float(self.missing_cfg["minimum_observed_herd_ratio"])
        )
        persistence = int(self.missing_cfg["persistence_frames"])
        minimum_confidence = float(self.missing_cfg["minimum_confidence"])

        for track_id in self.reference_tracks:
            if track_id in current:
                self.absence_frames[track_id] = 0
                continue
            if not (eligible_hour and reliable_frame):
                continue
            self.absence_frames[track_id] += 1
            absent_frames = self.absence_frames[track_id]
            if absent_frames < persistence:
                continue
            persistence_score = min(1.0, absent_frames / (persistence + 2))
            confidence = (
                minimum_confidence
                + 0.20 * persistence_score
                + 0.20 * min(1.0, observed_ratio)
            )
            predictions.append(
                {
                    "type": "animal_missing",
                    "track_id": track_id,
                    "frame": observation["frame"],
                    "timestamp": observation["timestamp"],
                    "confidence": round(min(0.99, confidence), 3),
                    "evidence": {
                        "consecutive_reliable_absences": absent_frames,
                        "observed_herd_ratio": round(observed_ratio, 3),
                        "observed_count": observed_count,
                        "reference_track_count": expected,
                        "reference_source": "visual_track_history",
                    },
                }
            )
        return predictions

    @staticmethod
    def _appearance_rgb(detection):
        appearance = detection.get("appearance") or {}
        value = appearance.get("foreground_rgb")
        if not isinstance(value, list) or len(value) != 3:
            return None
        return [float(channel) for channel in value]

    @staticmethod
    def _mean_rgb(values):
        return [
            sum(value[channel] for value in values) / len(values)
            for channel in range(3)
        ]

    def _death_predictions(self, observation, current):
        if not self.death_cfg.get("enabled", True):
            return []
        predictions = []
        minimum_span_frames = int(
            self.death_cfg.get("minimum_stationary_frames", 48)
        )
        minimum_observations = int(
            self.death_cfg.get("minimum_observed_frames", 34)
        )
        maximum_span = float(self.death_cfg.get("maximum_position_span_px", 14.0))
        baseline_samples = int(self.death_cfg.get("appearance_baseline_samples", 6))
        recent_samples = int(self.death_cfg.get("appearance_recent_samples", 3))
        minimum_color_change = float(
            self.death_cfg.get("minimum_color_change_rgb", 28.0)
        )
        minimum_confidence = float(self.death_cfg.get("minimum_confidence", 0.7))

        for track_id, detection in current.items():
            history = self.death_histories[track_id]
            history.append(
                {
                    "frame": int(observation["frame"]),
                    "center": detection["center_xy"],
                    "confidence": float(detection["confidence"]),
                    "appearance": self._appearance_rgb(detection),
                }
            )
            if track_id in self.confirmed_deaths or len(history) < minimum_observations:
                continue
            stationary_history = []
            position_span = 0.0
            for item in reversed(history):
                candidate = [item, *stationary_history]
                centers = [entry["center"] for entry in candidate]
                candidate_span = math.hypot(
                    max(center[0] for center in centers)
                    - min(center[0] for center in centers),
                    max(center[1] for center in centers)
                    - min(center[1] for center in centers),
                )
                if candidate_span > maximum_span:
                    break
                stationary_history = candidate
                position_span = candidate_span
            if len(stationary_history) < minimum_observations:
                continue
            if (
                stationary_history[-1]["frame"]
                - stationary_history[0]["frame"]
                < minimum_span_frames
            ):
                continue
            appearance_values = [
                item["appearance"]
                for item in stationary_history
                if item["appearance"] is not None
            ]
            if len(appearance_values) < baseline_samples + recent_samples:
                continue
            baseline = self._mean_rgb(appearance_values[:baseline_samples])
            recent = self._mean_rgb(appearance_values[-recent_samples:])
            color_change = math.sqrt(
                sum((recent[index] - baseline[index]) ** 2 for index in range(3))
            )
            if color_change < minimum_color_change:
                continue
            detector_confidence = sum(
                item["confidence"] for item in stationary_history
            ) / len(stationary_history)
            stationary_score = max(0.0, 1.0 - position_span / maximum_span)
            color_score = min(1.0, color_change / (minimum_color_change * 2))
            confidence = (
                detector_confidence * 0.35
                + stationary_score * 0.30
                + color_score * 0.35
            )
            if confidence < minimum_confidence:
                continue
            self.confirmed_deaths.add(track_id)
            predictions.append(
                {
                    "type": "animal_death",
                    "track_id": track_id,
                    "frame": observation["frame"],
                    "timestamp": observation["timestamp"],
                    "confidence": round(min(0.99, confidence), 3),
                    "evidence": {
                        "stationary_span_frames": (
                            stationary_history[-1]["frame"]
                            - stationary_history[0]["frame"]
                        ),
                        "observed_stationary_frames": len(stationary_history),
                        "position_span_px": round(position_span, 2),
                        "appearance_baseline_rgb": [round(value, 2) for value in baseline],
                        "appearance_recent_rgb": [round(value, 2) for value in recent],
                        "appearance_change_rgb": round(color_change, 2),
                        "appearance_source": "raw_bbox_pixels",
                    },
                }
            )
        return predictions

    def _parturition_predictions(self, observation, current):
        if not self.parturition_cfg.get("enabled", True):
            return []
        predictions = []
        frame = int(observation["frame"])
        prepartum_window = int(
            self.parturition_cfg.get("prepartum_window_frames", 4)
        )
        minimum_prior_frames = int(
            self.parturition_cfg.get("minimum_prior_frames", 12)
        )
        maximum_mother_step = float(
            self.parturition_cfg.get("maximum_mother_step_px", 32.0)
        )
        minimum_isolation = float(
            self.parturition_cfg.get("minimum_isolation_px", 20.0)
        )
        maximum_calf_distance = float(
            self.parturition_cfg.get("maximum_calf_distance_px", 20.0)
        )
        persistence = int(
            self.parturition_cfg.get("minimum_persistence_frames", 3)
        )
        association_window = int(
            self.parturition_cfg.get("new_track_window_frames", 4)
        )
        minimum_confidence = float(
            self.parturition_cfg.get("minimum_confidence", 0.68)
        )

        new_tracks = set(current) - self.known_tracks
        if self.processed_frames >= minimum_prior_frames:
            for new_track_id in sorted(new_tracks):
                existing = [track_id for track_id in current if track_id in self.known_tracks]
                if not existing:
                    continue
                mother_id = min(
                    existing,
                    key=lambda track_id: self._distance(
                        current[new_track_id]["center_xy"],
                        current[track_id]["center_xy"],
                    ),
                )
                history = list(self.histories[mother_id])[-prepartum_window:]
                if len(history) < prepartum_window:
                    continue
                steps = [
                    self._distance(history[index - 1]["center"], history[index]["center"])
                    for index in range(1, len(history))
                ]
                if steps and max(steps) > maximum_mother_step:
                    continue
                neighbors = [
                    self._distance(
                        current[mother_id]["center_xy"], detection["center_xy"]
                    )
                    for track_id, detection in current.items()
                    if track_id not in {mother_id, new_track_id}
                ]
                nearest_other = min(neighbors) if neighbors else float("inf")
                if nearest_other < minimum_isolation:
                    continue
                initial_distance = self._distance(
                    current[new_track_id]["center_xy"],
                    current[mother_id]["center_xy"],
                )
                if initial_distance > maximum_calf_distance:
                    continue
                self.birth_candidates[new_track_id] = {
                    "mother_id": mother_id,
                    "first_frame": frame,
                    "proximity_hits": 1,
                    "maximum_mother_step_px": max(steps, default=0.0),
                    "nearest_other_px": nearest_other,
                    "distances": [initial_distance],
                }

        for calf_id, candidate in list(self.birth_candidates.items()):
            if candidate.get("confirmed"):
                continue
            if frame - candidate["first_frame"] > association_window:
                del self.birth_candidates[calf_id]
                continue
            mother_id = candidate["mother_id"]
            if calf_id not in current or mother_id not in current:
                candidate["proximity_hits"] = 0
                continue
            if frame != candidate["first_frame"]:
                distance = self._distance(
                    current[calf_id]["center_xy"],
                    current[mother_id]["center_xy"],
                )
                candidate["distances"].append(distance)
                candidate["proximity_hits"] = (
                    candidate["proximity_hits"] + 1
                    if distance <= maximum_calf_distance
                    else 0
                )
            if candidate["proximity_hits"] < persistence:
                continue
            key = (mother_id, calf_id)
            if key in self.confirmed_parturitions:
                continue
            mean_detector_confidence = (
                float(current[mother_id]["confidence"])
                + float(current[calf_id]["confidence"])
            ) / 2
            movement_score = max(
                0.0,
                1.0 - candidate["maximum_mother_step_px"] / maximum_mother_step,
            )
            proximity_score = max(
                0.0,
                1.0 - max(candidate["distances"]) / maximum_calf_distance,
            )
            confidence = (
                mean_detector_confidence * 0.40
                + movement_score * 0.25
                + proximity_score * 0.35
            )
            if confidence < minimum_confidence:
                continue
            self.confirmed_parturitions.add(key)
            candidate["confirmed"] = True
            predictions.append(
                {
                    "type": "animal_parturition",
                    "track_id": mother_id,
                    "related_track_id": calf_id,
                    "frame": observation["frame"],
                    "timestamp": observation["timestamp"],
                    "confidence": round(min(0.99, confidence), 3),
                    "evidence": {
                        "calf_track_id": calf_id,
                        "new_track_first_frame": candidate["first_frame"],
                        "prepartum_window_frames": prepartum_window,
                        "maximum_mother_step_px": round(
                            candidate["maximum_mother_step_px"], 2
                        ),
                        "nearest_other_px": (
                            None
                            if math.isinf(candidate["nearest_other_px"])
                            else round(candidate["nearest_other_px"], 2)
                        ),
                        "proximity_persistence_frames": candidate["proximity_hits"],
                        "maximum_calf_distance_observed_px": round(
                            max(candidate["distances"]), 2
                        ),
                    },
                }
            )
        return predictions

    def analyze(self, observations):
        predictions = []
        for observation in observations:
            self.processed_frames += 1
            current = {
                int(item["track_id"]): item for item in observation["detections"]
            }
            self._update_population_reference(observation, current)
            predictions.extend(self._stationary_predictions(observation, current))
            predictions.extend(self._death_predictions(observation, current))
            predictions.extend(self._parturition_predictions(observation, current))
            predictions.extend(self._missing_predictions(observation, current))
            self.known_tracks.update(current)
        death_tracks = {
            prediction["track_id"]
            for prediction in predictions
            if prediction["type"] == "animal_death"
        }
        return [
            prediction
            for prediction in predictions
            if not (
                prediction["type"] == "animal_fallen"
                and prediction["track_id"] in death_tracks
            )
        ]
