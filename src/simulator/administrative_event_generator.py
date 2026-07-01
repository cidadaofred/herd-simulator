class AdministrativeEventGenerator:
    """Agenda mudanças conhecidas no inventário; não produz alertas."""

    SUPPORTED_TYPES = {"animal_sale", "animal_entry", "animal_transfer"}

    def __init__(self, config: dict):
        cfg = config.get("administrative_events", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.schedule = cfg.get("schedule", [])

    def events_at(self, day: int, hour: int):
        if not self.enabled:
            return []
        return [
            definition
            for definition in self.schedule
            if int(definition["day"]) == day and int(definition["hour"]) == hour
        ]
