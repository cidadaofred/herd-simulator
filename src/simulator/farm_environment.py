from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Tuple


@dataclass
class Paddock:
    name: str
    bbox: Tuple[int, int, int, int]
    pasture_quality: float
    carrying_capacity: int
    natural_shadow: float = 0.0
    accessible: bool = True
    has_water: bool = False
    has_shadow: bool = False


@dataclass
class WaterSource:
    id: str
    paddock_id: str
    position: Tuple[int, int]
    capacity_liters: float
    level_liters: float
    evaporation_lph_per_c: float
    minimum_operational_level: float = 50.0

    @property
    def available(self):
        return self.level_liters > self.minimum_operational_level


@dataclass
class Shelter:
    id: str
    paddock_id: str
    position: Tuple[int, int]
    capacity_animals: int
    size: Tuple[int, int]


class FarmEnvironment:
    def __init__(self, width: int, height: int, config: dict):
        self.width = width
        self.height = height
        self.config = config
        self.paddocks: Dict[str, Paddock] = {}
        for item in config["paddocks"]:
            self.paddocks[item["id"]] = Paddock(
                name=item["id"],
                bbox=tuple(item["bbox"]),
                pasture_quality=float(item["initial_pasture_quality"]),
                carrying_capacity=int(item["carrying_capacity"]),
                natural_shadow=float(item.get("natural_shadow", 0.0)),
                accessible=item.get("accessible", True),
            )
        self.water_sources = {
            item["id"]: WaterSource(
                id=item["id"],
                paddock_id=item["paddock_id"],
                position=tuple(item["position"]),
                capacity_liters=float(item["capacity_liters"]),
                level_liters=float(item["initial_level_liters"]),
                evaporation_lph_per_c=float(item.get("evaporation_lph_per_c", 0.0)),
                minimum_operational_level=float(
                    item.get("minimum_operational_level", 50.0)
                ),
            )
            for item in config.get("water_sources", [])
        }
        self.shelters = {
            item["id"]: Shelter(
                id=item["id"],
                paddock_id=item["paddock_id"],
                position=tuple(item["position"]),
                capacity_animals=int(item["capacity_animals"]),
                size=tuple(item.get("size", [150, 50])),
            )
            for item in config.get("shelters", [])
        }
        self.graph = {name: {} for name in self.paddocks}
        for connection in config.get("connections", []):
            origin, destination = connection["from"], connection["to"]
            distance = float(connection["distance"])
            self.graph[origin][destination] = distance
            if connection.get("bidirectional", True):
                self.graph[destination][origin] = distance
        for paddock in self.paddocks.values():
            paddock.has_water = any(
                source.paddock_id == paddock.name for source in self.water_sources.values()
            )
            paddock.has_shadow = any(
                shelter.paddock_id == paddock.name for shelter in self.shelters.values()
            ) or paddock.natural_shadow > 0

    def get_paddock_bbox(self, paddock_name: str):
        return self.paddocks[paddock_name].bbox

    def get_paddock_center(self, paddock_name: str):
        x1, y1, x2, y2 = self.get_paddock_bbox(paddock_name)
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def water_sources_in(self, paddock_name: str, available_only=True):
        return [
            source
            for source in self.water_sources.values()
            if source.paddock_id == paddock_name
            and (source.available or not available_only)
        ]

    def available_water_sources(self):
        return [source for source in self.water_sources.values() if source.available]

    def shelters_in(self, paddock_name: str):
        return [
            shelter
            for shelter in self.shelters.values()
            if shelter.paddock_id == paddock_name
        ]

    @staticmethod
    def shelter_bbox(shelter: Shelter):
        width, height = shelter.size
        center_x, center_y = shelter.position
        return (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )

    def refuge_center(self, paddock_name: str):
        shelters = self.shelters_in(paddock_name)
        if shelters:
            return shelters[0].position
        x1, y1, x2, y2 = self.get_paddock_bbox(paddock_name)
        if self.paddocks[paddock_name].natural_shadow > 0:
            return self.natural_shadow_center(paddock_name)
        return self.get_paddock_center(paddock_name)

    def natural_shadow_center(self, paddock_name: str):
        x1, y1, x2, y2 = self.get_paddock_bbox(paddock_name)
        return (x1 + (x2 - x1) * 0.22, y1 + (y2 - y1) * 0.78)

    def shelter_capacity(self, paddock_name: str):
        return sum(
            shelter.capacity_animals for shelter in self.shelters_in(paddock_name)
        )

    def refuge_kind(self, paddock_name: str):
        if self.shelters_in(paddock_name):
            return "shelter"
        if self.paddocks[paddock_name].natural_shadow > 0:
            return "natural_shadow"
        return "open_field"

    def best_refuge_paddock(self, origin: str):
        candidates = [
            paddock
            for paddock in self.paddocks.values()
            if paddock.accessible
            and paddock.has_shadow
            and self.is_reachable(origin, paddock.name)
        ]
        if not candidates:
            return origin
        return min(
            candidates,
            key=lambda paddock: self.route_distance(origin, paddock.name),
        ).name

    def route_distance(self, origin: str, destination: str):
        if origin == destination:
            return 0.0
        queue = [(0.0, origin)]
        visited = set()
        while queue:
            distance, current = heappop(queue)
            if current == destination:
                return distance
            if current in visited:
                continue
            visited.add(current)
            for neighbor, edge_distance in self.graph.get(current, {}).items():
                if neighbor not in visited:
                    heappush(queue, (distance + edge_distance, neighbor))
        return float("inf")

    def is_reachable(self, origin: str, destination: str):
        return self.route_distance(origin, destination) != float("inf")

    def update_water(self, temperature, duration_hours, drinkers_by_source, evaporation_multiplier=1.0):
        liters_per_drinking_frame = float(
            self.config.get("liters_per_drinking_frame", 8.0)
        )
        for source in self.water_sources.values():
            heat_degrees = max(0.0, temperature - 10.0)
            evaporation = (
                source.evaporation_lph_per_c
                * heat_degrees
                * duration_hours
                * evaporation_multiplier
            )
            consumption = (
                drinkers_by_source.get(source.id, 0)
                * liters_per_drinking_frame
            )
            source.level_liters = max(
                0.0, source.level_liters - evaporation - consumption
            )

    def pasture_snapshot(self):
        return {
            name: round(paddock.pasture_quality, 4)
            for name, paddock in self.paddocks.items()
        }

    def resource_snapshot(self):
        return {
            "water_sources": {
                source.id: {
                    "paddock_id": source.paddock_id,
                    "level_liters": round(source.level_liters, 2),
                    "capacity_liters": source.capacity_liters,
                    "available": source.available,
                }
                for source in self.water_sources.values()
            },
            "shelters": {
                shelter.id: {
                    "paddock_id": shelter.paddock_id,
                    "capacity_animals": shelter.capacity_animals,
                }
                for shelter in self.shelters.values()
            },
        }
