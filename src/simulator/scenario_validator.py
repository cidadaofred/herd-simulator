class ScenarioValidator:
    """Valida o cenário antes que qualquer frame seja produzido."""

    @classmethod
    def validate(cls, config: dict):
        errors = []
        farm = config.get("farm", {})
        environment = config.get("environment", {})
        paddocks = environment.get("paddocks", [])
        width, height = farm.get("width", 0), farm.get("height", 0)

        ids = [paddock.get("id") for paddock in paddocks]
        if not paddocks:
            errors.append("environment.paddocks deve possuir ao menos um piquete.")
        if len(ids) != len(set(ids)):
            errors.append("Os IDs dos piquetes devem ser únicos.")
        paddock_ids = set(ids)

        for paddock in paddocks:
            bbox = paddock.get("bbox", [])
            if len(bbox) != 4:
                errors.append(f"Piquete {paddock.get('id')} possui bbox inválido.")
                continue
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(f"Piquete {paddock.get('id')} ultrapassa o mapa.")
            if paddock.get("carrying_capacity", 0) <= 0:
                errors.append(f"Piquete {paddock.get('id')} precisa de capacidade positiva.")

        def validate_resources(resource_name):
            resource_ids = []
            for resource in environment.get(resource_name, []):
                resource_ids.append(resource.get("id"))
                paddock_id = resource.get("paddock_id")
                if paddock_id not in paddock_ids:
                    errors.append(
                        f"{resource_name}.{resource.get('id')} referencia piquete inexistente."
                    )
                    continue
                position = resource.get("position", [])
                if len(position) != 2:
                    errors.append(f"{resource_name}.{resource.get('id')} sem posição válida.")
                    continue
                paddock = next(item for item in paddocks if item["id"] == paddock_id)
                x1, y1, x2, y2 = paddock["bbox"]
                if not (x1 <= position[0] <= x2 and y1 <= position[1] <= y2):
                    errors.append(
                        f"{resource_name}.{resource.get('id')} está fora de {paddock_id}."
                    )
            if len(resource_ids) != len(set(resource_ids)):
                errors.append(f"Os IDs em {resource_name} devem ser únicos.")

        validate_resources("water_sources")
        validate_resources("shelters")
        if not environment.get("water_sources"):
            errors.append("O cenário precisa de ao menos uma fonte de água.")

        for connection in environment.get("connections", []):
            if connection.get("from") not in paddock_ids or connection.get("to") not in paddock_ids:
                errors.append(f"Conexão inválida: {connection}.")
            if connection.get("distance", 0) <= 0:
                errors.append(f"Conexão precisa de distância positiva: {connection}.")

        home = config.get("behavior_planner", {}).get("initial_home_paddock")
        if home not in paddock_ids:
            errors.append("behavior_planner.initial_home_paddock não existe no ambiente.")
        for lot in config.get("social_behavior", {}).get("lots", []):
            if lot.get("enabled", True) and lot.get("preferred_shelter") not in paddock_ids:
                errors.append(
                    f"Lote {lot.get('id')} referencia piquete preferencial inexistente."
                )

        adjacency = {paddock_id: set() for paddock_id in paddock_ids}
        for connection in environment.get("connections", []):
            origin, destination = connection.get("from"), connection.get("to")
            if origin in adjacency and destination in adjacency:
                adjacency[origin].add(destination)
                if connection.get("bidirectional", True):
                    adjacency[destination].add(origin)
        reachable = set()
        pending = [home] if home in adjacency else []
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(adjacency[current] - reachable)
        water_paddocks = {
            source.get("paddock_id")
            for source in environment.get("water_sources", [])
            if source.get("initial_level_liters", 0)
            > source.get("minimum_operational_level", 50)
        }
        if reachable and not reachable.intersection(water_paddocks):
            errors.append("Nenhuma fonte de água operacional é alcançável a partir do home.")

        hours = config.get("simulation", {}).get("frame_hours", [])
        weather = config.get("weather", {})
        for hour in hours:
            if str(hour) not in weather.get("temperature_by_hour", {}):
                errors.append(f"Temperatura não configurada para {hour}h.")
            if str(hour) not in weather.get("wind_by_hour", {}):
                errors.append(f"Vento não configurado para {hour}h.")

        num_cattle = config.get("simulation", {}).get("num_bovinos", 0)
        campaign = config.get("campaign", {})
        initially_available_ids = set(range(1, int(num_cattle) + 1))
        initially_available_ids.update(
            int(item) for item in campaign.get("checkpoint_animal_ids", [])
        )
        entry_at = {}
        sale_at = {}
        for administrative in config.get("administrative_events", {}).get(
            "schedule", []
        ):
            specifications = administrative.get("animals", [])
            administrative_ids = list(administrative.get("animal_ids", [])) + [
                item.get("id") for item in specifications
            ]
            moment = (
                int(administrative.get("day", 0)),
                int(administrative.get("hour", -1)),
            )
            if administrative.get("type") == "animal_entry":
                for animal_id in administrative_ids:
                    if isinstance(animal_id, int):
                        entry_at[animal_id] = min(
                            moment, entry_at.get(animal_id, moment)
                        )
            elif administrative.get("type") == "animal_sale":
                for animal_id in administrative_ids:
                    if isinstance(animal_id, int):
                        sale_at[animal_id] = min(moment, sale_at.get(animal_id, moment))

        for occurrence in config.get("events", {}).get("schedule", []):
            animal_id = occurrence.get("animal_id")
            occurrence_at = (
                int(occurrence.get("day", 0)),
                int(occurrence.get("hour", -1)),
            )
            available = (
                animal_id in initially_available_ids
                or (
                    animal_id in entry_at
                    and entry_at[animal_id] <= occurrence_at
                )
            )
            already_sold = (
                animal_id in sale_at and sale_at[animal_id] <= occurrence_at
            )
            if animal_id is not None and (not available or already_sold):
                errors.append(
                    f"Evento {occurrence.get('id')} referencia animal inexistente."
                )
            if occurrence.get("day", 0) < 1 or occurrence.get("hour") not in hours:
                errors.append(f"Evento {occurrence.get('id')} possui data/hora inválida.")

        supported_administrative = {
            "animal_sale",
            "animal_entry",
            "animal_transfer",
        }
        for event in config.get("administrative_events", {}).get("schedule", []):
            if event.get("type") not in supported_administrative:
                errors.append(f"Evento administrativo {event.get('id')} possui tipo inválido.")
            specifications = event.get("animals", [])
            animal_ids = list(event.get("animal_ids", [])) + [
                item.get("id") for item in specifications
            ]
            if not animal_ids:
                errors.append(f"Evento administrativo {event.get('id')} não possui animais.")
            branch_mode = config.get("campaign", {}).get("mode") == "branch"
            invalid_ids = [
                animal_id
                for animal_id in animal_ids
                if not isinstance(animal_id, int)
                or animal_id < 1
                or (
                    animal_id > num_cattle
                    and event.get("type") != "animal_entry"
                    and not branch_mode
                )
            ]
            if invalid_ids:
                errors.append(
                    f"Evento administrativo {event.get('id')} referencia IDs inválidos: "
                    f"{invalid_ids}."
                )
            if event.get("day", 0) < 1 or event.get("hour") not in hours:
                errors.append(
                    f"Evento administrativo {event.get('id')} possui data/hora inválida."
                )
            target = event.get("target_paddock")
            if event.get("type") == "animal_transfer" and target not in paddock_ids:
                errors.append(
                    f"Evento administrativo {event.get('id')} referencia piquete inválido."
                )

        if campaign.get("enabled", False):
            if campaign.get("mode", "new") not in {"new", "resume", "branch"}:
                errors.append("campaign.mode deve ser new, resume ou branch.")
            try:
                from datetime import date

                date.fromisoformat(campaign.get("start_date", ""))
            except (TypeError, ValueError):
                errors.append("campaign.start_date deve usar o formato AAAA-MM-DD.")

        if errors:
            raise ValueError("Cenário inválido:\n- " + "\n- ".join(errors))
        return True
