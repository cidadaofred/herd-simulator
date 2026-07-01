class EventGenerator:
    """Ativa ocorrências tipadas e determinísticas definidas pelo cenário."""

    def __init__(self, config: dict):
        self.enabled = config["events"].get("enabled", False)
        self.schedule = config["events"].get("schedule", [])

    @staticmethod
    def _timestamp(day: int, hour: int):
        return (day - 1) * 24 + hour

    def active_events(self, day: int, hour: int):
        if not self.enabled:
            return []
        current = self._timestamp(day, hour)
        active = []
        for definition in self.schedule:
            start = self._timestamp(definition["day"], definition["hour"])
            if definition.get("permanent", False):
                is_active = current >= start
            else:
                end = start + float(definition.get("duration_hours", 1))
                is_active = start <= current < end
            if is_active:
                active.append(
                    {
                        "id": definition["id"],
                        "type": definition["type"],
                        "animal_id": definition.get("animal_id"),
                        "start_day": definition["day"],
                        "start_hour": definition["hour"],
                        "parameters": definition.get("parameters", {}),
                    }
                )
        return active

    def occurrence_ground_truth(self, day: int, hour: int):
        return self.active_events(day, hour)
