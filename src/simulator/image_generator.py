import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


class ImageGenerator:
    """Renderiza a captura bruta do drone e a saída do detector."""

    CATEGORY_COLORS = {
        "terneiro": (255, 215, 70),
        "vaca_lactante": (255, 90, 195),
        "vaca_adulta": (245, 245, 245),
        "novilha": (50, 220, 255),
    }
    CATEGORY_IDS = {
        "terneiro": 0,
        "vaca_lactante": 1,
        "vaca_adulta": 2,
        "novilha": 3,
    }

    def __init__(
        self,
        farm,
        output_dir: Path,
        base_map_dir: Path,
        pasture_model=None,
        rendering_config=None,
    ):
        self.farm = farm
        self.pasture_model = pasture_model
        self.config = rendering_config or {}
        self.output_dir = output_dir.resolve()
        self.base_map_dir = base_map_dir.resolve()
        data_dir = self.output_dir.parent
        self.raw_dir = data_dir / "raw_drone_frames"
        self.labels_dir = data_dir / "labels_yolo"
        for directory in (
            self.output_dir,
            self.base_map_dir,
            self.raw_dir,
            self.labels_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.current_base_map_path = None
        self.current_day = 0
        self.last_positions = {}
        self.last_visibility_ground_truth = {}

    @staticmethod
    def _color_by_pasture_quality(quality: float):
        dry = (151, 116, 62)
        green = (59, 135, 62)
        quality = max(0.0, min(1.0, quality))
        return tuple(
            int(dry[index] * (1.0 - quality) + green[index] * quality)
            for index in range(3)
        )

    def _textured_zone(self, image, bbox, quality, rng):
        draw = ImageDraw.Draw(image)
        base = self._color_by_pasture_quality(quality)
        draw.rectangle(bbox, fill=base)
        x1, y1, x2, y2 = (int(value) for value in bbox)
        density = self.config.get("texture_density", 0.015)
        count = int((x2 - x1) * (y2 - y1) * density)
        for _ in range(count):
            x = rng.randint(x1, x2)
            y = rng.randint(y1, y2)
            delta = rng.randint(-22, 22)
            color = tuple(max(0, min(255, channel + delta)) for channel in base)
            length = rng.choice((1, 1, 2, 3))
            draw.line((x, y, x + length, y), fill=color)

    def prepare_day(self, day: int):
        self.current_day = day
        self.current_base_map_path = self.base_map_dir / f"farm_base_day_{day:02d}.jpg"
        self.draw_base_map(self.current_base_map_path, day)

    def draw_base_map(self, path: Path, day: int):
        rng = random.Random(7000 + day)
        image = Image.new(
            "RGB",
            (self.farm.width, self.farm.height),
            (145, 132, 101),
        )
        draw = ImageDraw.Draw(image)

        # Estrada rural central com pequenas variações de cascalho.
        drawn_connections = set()
        for origin, neighbors in self.farm.graph.items():
            for destination in neighbors:
                key = tuple(sorted((origin, destination)))
                if key in drawn_connections:
                    continue
                drawn_connections.add(key)
                draw.line(
                    (
                        *self.farm.get_paddock_center(origin),
                        *self.farm.get_paddock_center(destination),
                    ),
                    fill=(112, 101, 80),
                    width=14,
                )

        for name, paddock in self.farm.paddocks.items():
            x1, y1, x2, y2 = paddock.bbox
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            zone_boxes = {
                "NW": (x1, y1, center_x, center_y),
                "NE": (center_x, y1, x2, center_y),
                "SW": (x1, center_y, center_x, y2),
                "SE": (center_x, center_y, x2, y2),
            }
            for suffix, zone_bbox in zone_boxes.items():
                quality = (
                    self.pasture_model.zone_quality(f"{name}_{suffix}")
                    if self.pasture_model
                    else paddock.pasture_quality
                )
                self._textured_zone(image, zone_bbox, quality, rng)

            draw = ImageDraw.Draw(image)
            draw.rectangle(paddock.bbox, outline=(70, 70, 48), width=3)
            # Cerca com postes vistos de cima.
            for x in range(x1, x2 + 1, 28):
                draw.ellipse((x - 2, y1 - 2, x + 2, y1 + 2), fill=(55, 47, 36))
                draw.ellipse((x - 2, y2 - 2, x + 2, y2 + 2), fill=(55, 47, 36))
            for y in range(y1, y2 + 1, 28):
                draw.ellipse((x1 - 2, y - 2, x1 + 2, y + 2), fill=(55, 47, 36))
                draw.ellipse((x2 - 2, y - 2, x2 + 2, y + 2), fill=(55, 47, 36))

            for shelter in self.farm.shelters_in(name):
                width, height = shelter.size
                center_x, center_y = shelter.position
                roof = (
                    center_x - width // 2,
                    center_y - height // 2,
                    center_x + width // 2,
                    center_y + height // 2,
                )
                draw.rectangle((roof[0] + 7, roof[1] + 7, roof[2] + 9, roof[3] + 9), fill=(42, 49, 39))
                draw.rectangle(roof, fill=(97, 82, 61), outline=(55, 47, 37), width=2)
                for y in range(roof[1] + 5, roof[3], 8):
                    draw.line((roof[0] + 2, y, roof[2] - 2, y), fill=(117, 99, 72))

            tree_count = int(paddock.natural_shadow * 24)
            for index in range(tree_count):
                tree_x = x1 + 24 + (index * 53) % max(30, x2 - x1 - 48)
                tree_y = y1 + 24 + (index * 71) % max(30, y2 - y1 - 48)
                draw.ellipse(
                    (tree_x - 10, tree_y - 8, tree_x + 10, tree_y + 8),
                    fill=(44, 91, 43),
                    outline=(35, 66, 34),
                )

        image = ImageEnhance.Contrast(image).enhance(1.05)
        image.save(path, format="JPEG", quality=94)

    def _camera_offset(self, frame_index: int):
        limit = int(self.config.get("camera_jitter_px", 0))
        rng = random.Random(81000 + frame_index)
        return rng.randint(-limit, limit), rng.randint(-limit, limit)

    def _coat_color(self, cattle_id: int):
        colors = self.config.get("coat_colors", [[220, 210, 190]])
        return tuple(colors[(cattle_id * 7) % len(colors)])

    def _draw_dynamic_water(self, image, camera_offset):
        draw = ImageDraw.Draw(image)
        dx, dy = camera_offset
        for source in self.farm.water_sources.values():
            ratio = max(0.0, min(1.0, source.level_liters / source.capacity_liters))
            radius_x = max(10, int(28 + 38 * math.sqrt(ratio)))
            radius_y = max(7, int(radius_x * 0.72))
            center_x = source.position[0] + dx
            center_y = source.position[1] + dy
            pond = (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            )
            color = (48, 109, 145) if source.available else (104, 91, 61)
            draw.ellipse(pond, fill=color, outline=(38, 78, 99), width=3)
            if source.available:
                draw.arc(
                    (pond[0] + 8, pond[1] + 8, pond[2] - 8, pond[3] - 8),
                    200,
                    330,
                    fill=(105, 160, 183),
                    width=2,
                )

    def _animal_sprite(
        self,
        cattle_id,
        category,
        angle,
        hour,
        physical_state,
        decomposition_days=0,
        decomposition_change_day=5,
    ):
        body_width, body_height = self.config["animal_size_px"][category]
        if physical_state in {"fallen", "dead"}:
            body_width = int(body_width * 1.25)
            body_height = int(body_height * 1.45)
        canvas_size = 48
        sprite = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        coat = self._coat_color(cattle_id)
        if physical_state == "dead":
            if int(decomposition_days) >= int(decomposition_change_day):
                progress = min(
                    1.0,
                    (int(decomposition_days) - int(decomposition_change_day) + 1) / 2,
                )
                decay_color = (92, 72, 40)
                coat = tuple(
                    int(value * (1.0 - progress) + decay * progress)
                    for value, decay in zip(coat, decay_color)
                )
            else:
                gray = int(sum(coat) / 3)
                coat = tuple(int(value * 0.55 + gray * 0.45) for value in coat)
        left = (canvas_size - body_width) // 2
        top = (canvas_size - body_height) // 2
        draw.ellipse(
            (left, top, left + body_width, top + body_height),
            fill=(*coat, 255),
            outline=(35, 28, 24, 255),
        )
        head_size = max(3, body_height // 2)
        draw.ellipse(
            (
                left + body_width - 1,
                top + body_height // 2 - head_size // 2,
                left + body_width + head_size,
                top + body_height // 2 + head_size // 2,
            ),
            fill=(*coat, 255),
        )
        # Manchas individuais mantidas de forma determinística.
        if cattle_id % 3 == 0:
            draw.ellipse(
                (left + body_width // 3, top + 1, left + body_width // 2 + 2, top + body_height - 1),
                fill=(42, 36, 31, 210),
            )
        if physical_state == "dead" and int(decomposition_days) >= int(
            decomposition_change_day
        ):
            draw.ellipse(
                (
                    left + body_width // 5,
                    top + 2,
                    left + body_width // 2,
                    top + body_height - 2,
                ),
                fill=(58, 67, 37, 220),
            )
        if physical_state in {"fallen", "dead"}:
            draw.line(
                (left + 2, top - 3, left + body_width - 2, top - 3),
                fill=(*coat, 255),
                width=2,
            )
            draw.line(
                (left + 2, top + body_height + 3, left + body_width - 2, top + body_height + 3),
                fill=(*coat, 255),
                width=2,
            )
        return sprite.rotate(
            -math.degrees(angle),
            resample=Image.Resampling.BICUBIC,
            expand=False,
        )

    def _heading(self, animal):
        previous = self.last_positions.get(animal["id"])
        if previous:
            dx = animal["x"] - previous[0]
            dy = animal["y"] - previous[1]
            if abs(dx) + abs(dy) > 1:
                return math.atan2(dy, dx)
        return math.radians((animal["id"] * 47) % 360)

    @staticmethod
    def _sun_shadow_offset(hour: int):
        distance = max(2, int(abs(13 - hour) * 1.3 + 2))
        angle = math.radians(225 if hour <= 13 else 45)
        return int(math.cos(angle) * distance), int(math.sin(angle) * distance)

    def _render_cattle(self, image, positions, hour, camera_offset, visibility):
        shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        cattle_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_x, shadow_y = self._sun_shadow_offset(hour)
        dx, dy = camera_offset
        for animal in positions:
            angle = self._heading(animal)
            sprite = self._animal_sprite(
                animal["id"],
                animal["category"],
                angle,
                hour,
                animal.get("physical_state", "normal"),
                animal.get("decomposition_days", 0),
                animal.get("decomposition_color_change_day", 5),
            )
            x = int(animal["x"] + dx - sprite.width / 2)
            y = int(animal["y"] + dy - sprite.height / 2)
            alpha = sprite.getchannel("A")
            visibility_item = visibility[animal["id"]]
            if visibility_item["visibility_status"] == "partially_occluded":
                roof = visibility_item["occluding_bbox_xyxy"]
                alpha_draw = ImageDraw.Draw(alpha)
                alpha_draw.rectangle(
                    (
                        int(roof[0] + dx - x),
                        int(roof[1] + dy - y),
                        int(roof[2] + dx - x),
                        int(roof[3] + dy - y),
                    ),
                    fill=0,
                )
                sprite.putalpha(alpha)
            shadow = Image.new("RGBA", sprite.size, (15, 12, 10, 105))
            shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(1.2)))
            shadow_layer.alpha_composite(shadow, (x + shadow_x, y + shadow_y))
            cattle_layer.alpha_composite(sprite, (x, y))
        image = Image.alpha_composite(image.convert("RGBA"), shadow_layer)
        return Image.alpha_composite(image, cattle_layer).convert("RGB")

    def _animal_body_bbox(self, animal):
        body_width, body_height = self.config["animal_size_px"][animal["category"]]
        if animal.get("physical_state") in {"fallen", "dead"}:
            body_width = int(body_width * 1.25)
            body_height = int(body_height * 1.45)
        return (
            animal["x"] - body_width / 2,
            animal["y"] - body_height / 2,
            animal["x"] + body_width / 2,
            animal["y"] + body_height / 2,
        )

    @staticmethod
    def _rectangles_intersect(first, second):
        return not (
            first[2] <= second[0]
            or first[0] >= second[2]
            or first[3] <= second[1]
            or first[1] >= second[3]
        )

    @staticmethod
    def _intersection_area(first, second):
        width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
        return width * height

    @staticmethod
    def _largest_visible_rectangle(body_bbox, roof_bbox):
        x1, y1, x2, y2 = body_bbox
        rx1, ry1, rx2, ry2 = roof_bbox
        candidates = []
        if x1 < rx1:
            candidates.append((x1, y1, min(x2, rx1), y2))
        if x2 > rx2:
            candidates.append((max(x1, rx2), y1, x2, y2))
        if y1 < ry1:
            candidates.append((x1, y1, x2, min(y2, ry1)))
        if y2 > ry2:
            candidates.append((x1, max(y1, ry2), x2, y2))
        candidates = [
            candidate
            for candidate in candidates
            if candidate[2] > candidate[0] and candidate[3] > candidate[1]
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item[2] - item[0]) * (item[3] - item[1]),
        )

    def _classify_visibility(self, positions):
        occlusion_cfg = self.config.get("occlusion", {})
        enabled = occlusion_cfg.get("enabled", False)
        full_occlusion = occlusion_cfg.get("full_shelter_occlusion", True)
        partial_occlusion = occlusion_cfg.get("partial_shelter_occlusion", False)
        result = {}
        for animal in positions:
            classification = {
                "visible_from_drone": True,
                "visibility_status": "visible",
                "occlusion_reason": None,
                "occluding_resource_id": None,
                "visible_fraction": 1.0,
                "occluded_fraction": 0.0,
                "visible_bbox_xyxy": None,
                "occluding_bbox_xyxy": None,
            }
            if enabled:
                body_bbox = self._animal_body_bbox(animal)
                for shelter in self.farm.shelters_in(animal["paddock"]):
                    roof_bbox = self.farm.shelter_bbox(shelter)
                    body_area = max(
                        1.0,
                        (body_bbox[2] - body_bbox[0])
                        * (body_bbox[3] - body_bbox[1]),
                    )
                    occluded_fraction = min(
                        1.0,
                        self._intersection_area(body_bbox, roof_bbox) / body_area,
                    )
                    fully_covered = (
                        body_bbox[0] >= roof_bbox[0]
                        and body_bbox[1] >= roof_bbox[1]
                        and body_bbox[2] <= roof_bbox[2]
                        and body_bbox[3] <= roof_bbox[3]
                    )
                    if fully_covered and full_occlusion:
                        classification.update(
                            visible_from_drone=False,
                            visibility_status="fully_occluded",
                            occlusion_reason="shelter_roof",
                            occluding_resource_id=shelter.id,
                            visible_fraction=0.0,
                            occluded_fraction=1.0,
                            occluding_bbox_xyxy=list(roof_bbox),
                        )
                        break
                    if (
                        partial_occlusion
                        and self._rectangles_intersect(body_bbox, roof_bbox)
                    ):
                        visible_bbox = self._largest_visible_rectangle(
                            body_bbox, roof_bbox
                        )
                        classification.update(
                            visibility_status="partially_occluded",
                            occlusion_reason="shelter_edge",
                            occluding_resource_id=shelter.id,
                            visible_fraction=round(1.0 - occluded_fraction, 4),
                            occluded_fraction=round(occluded_fraction, 4),
                            visible_bbox_xyxy=(
                                list(visible_bbox) if visible_bbox else None
                            ),
                            occluding_bbox_xyxy=list(roof_bbox),
                        )
                        break
            result[animal["id"]] = classification
        return result

    def _apply_camera_effects(self, image, frame_index, hour):
        brightness = 0.88 + max(0, 1 - abs(13 - hour) / 7) * 0.16
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Color(image).enhance(0.92)
        blur = float(self.config.get("blur_radius", 0))
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        rng = random.Random(91000 + frame_index)
        pixels = image.load()
        for _ in range(int(self.config.get("sensor_noise_points", 0))):
            x = rng.randrange(image.width)
            y = rng.randrange(image.height)
            original = pixels[x, y]
            delta = rng.randint(-12, 12)
            pixels[x, y] = tuple(
                max(0, min(255, channel + delta)) for channel in original
            )
        return image

    def _zone_labels(self, draw, metadata, camera_offset):
        if not self.config.get("show_zone_labels_in_processed", True):
            return
        dx, dy = camera_offset
        qualities = metadata.get("pasture_zone_quality", {})
        for paddock_name, paddock in self.farm.paddocks.items():
            x1, y1, x2, y2 = paddock.bbox
            points = {
                "NW": (x1 + 8, y1 + 8),
                "NE": ((x1 + x2) // 2 + 8, y1 + 8),
                "SW": (x1 + 8, (y1 + y2) // 2 + 8),
                "SE": ((x1 + x2) // 2 + 8, (y1 + y2) // 2 + 8),
            }
            for suffix, point in points.items():
                quality = qualities.get(f"{paddock_name}_{suffix}")
                if quality is not None:
                    draw.text(
                        (point[0] + dx, point[1] + dy),
                        f"{paddock_name}-{suffix} {quality:.2f}",
                        fill=(238, 245, 225),
                        stroke_width=2,
                        stroke_fill=(25, 40, 25),
                    )

    def _detect(self, positions, frame_index, camera_offset, visibility):
        cfg = self.config["detection"]
        rng = random.Random(101000 + frame_index)
        detections = []
        labels = []
        dx, dy = camera_offset
        for animal in positions:
            visibility_item = visibility[animal["id"]]
            occluded_fraction = visibility_item["occluded_fraction"]
            neighbors = sum(
                1
                for other in positions
                if other["id"] != animal["id"]
                and math.hypot(other["x"] - animal["x"], other["y"] - animal["y"])
                < 18
            )
            miss_probability = cfg["miss_probability"] + min(0.16, neighbors * 0.018)
            miss_probability += (
                occluded_fraction
                * self.config.get("occlusion", {}).get(
                    "partial_miss_probability_max", 0.0
                )
            )
            if rng.random() < miss_probability:
                continue
            category = animal["category"]
            body_width, body_height = self.config["animal_size_px"][category]
            jitter = int(cfg["localization_jitter_px"])
            center_x = animal["x"] + dx + rng.randint(-jitter, jitter)
            center_y = animal["y"] + dy + rng.randint(-jitter, jitter)
            box_width = body_width * 1.8 + 8
            box_height = body_width * 1.8 + 8
            visible_bbox = visibility_item.get("visible_bbox_xyxy")
            if visible_bbox:
                padding = 3
                bbox = (
                    int(visible_bbox[0] + dx - padding),
                    int(visible_bbox[1] + dy - padding),
                    int(visible_bbox[2] + dx + padding),
                    int(visible_bbox[3] + dy + padding),
                )
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                box_width = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]
            else:
                bbox = (
                    int(center_x - box_width / 2),
                    int(center_y - box_height / 2),
                    int(center_x + box_width / 2),
                    int(center_y + box_height / 2),
                )
            confidence = rng.uniform(cfg["confidence_min"], cfg["confidence_max"])
            confidence -= min(0.12, neighbors * 0.012)
            confidence -= (
                occluded_fraction
                * self.config.get("occlusion", {}).get(
                    "partial_confidence_penalty_max", 0.0
                )
            )
            confidence = max(0.01, min(0.99, confidence))
            detections.append(
                {
                    "id": animal["id"],
                    "category": category,
                    "lot_id": animal.get("lot_id"),
                    "confidence": round(confidence, 3),
                    "bbox_xyxy": list(bbox),
                    "visibility_status": visibility_item["visibility_status"],
                    "visible_fraction": visibility_item["visible_fraction"],
                }
            )
            normalized_x = center_x / self.farm.width
            normalized_y = center_y / self.farm.height
            normalized_width = box_width / self.farm.width
            normalized_height = box_height / self.farm.height
            labels.append(
                f"{self.CATEGORY_IDS[category]} {normalized_x:.6f} "
                f"{normalized_y:.6f} {normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )
        return detections, labels

    def _draw_processed_overlay(self, image, detections, metadata, camera_offset):
        image = ImageEnhance.Contrast(image).enhance(1.08)
        image = ImageEnhance.Sharpness(image).enhance(1.3)
        draw = ImageDraw.Draw(image)
        self._zone_labels(draw, metadata, camera_offset)
        occupied_labels = []

        def overlaps(candidate):
            return any(
                not (
                    candidate[2] < existing[0]
                    or candidate[0] > existing[2]
                    or candidate[3] < existing[1]
                    or candidate[1] > existing[3]
                )
                for existing in occupied_labels
            )

        for detection in detections:
            color = self.CATEGORY_COLORS[detection["category"]]
            bbox = tuple(detection["bbox_xyxy"])
            draw.rectangle(bbox, outline=color, width=2)
            label = f"B{detection['id']} {detection['confidence']:.2f}"
            candidates = [
                (bbox[0], bbox[1] - 13, bbox[0] + 54, bbox[1]),
                (bbox[0], bbox[3], bbox[0] + 54, bbox[3] + 13),
                (bbox[2], bbox[1], bbox[2] + 54, bbox[1] + 13),
            ]
            text_box = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate[0] >= 0
                    and candidate[1] >= 50
                    and candidate[2] < self.farm.width
                    and candidate[3] < self.farm.height
                    and not overlaps(candidate)
                ),
                None,
            )
            if text_box is None:
                continue
            occupied_labels.append(text_box)
            draw.rectangle(text_box, fill=(15, 25, 28))
            draw.text((text_box[0] + 2, text_box[1] + 1), label, fill=color)

        draw.rectangle((0, 0, self.farm.width, 50), fill=(15, 25, 29))
        draw.text(
            (18, 10),
            (
                f"DRONE PRE-PROCESSADO | Dia {metadata['day']:02d} | "
                f"{metadata['hora']:02d}:00 | detectados {len(detections)}"
            ),
            fill=(235, 240, 240),
        )
        return image

    def generate_frame_from_positions(
        self,
        frame_index: int,
        cattle_positions: list[dict],
        events: list[str],
        metadata: dict,
    ):
        if self.current_base_map_path is None:
            raise RuntimeError("prepare_day deve ser chamado antes de gerar frames.")

        camera_offset = self._camera_offset(frame_index)
        visibility = self._classify_visibility(cattle_positions)
        visible_positions = [
            animal
            for animal in cattle_positions
            if visibility[animal["id"]]["visible_from_drone"]
        ]
        fully_occluded = [
            cattle_id
            for cattle_id, item in visibility.items()
            if item["visibility_status"] == "fully_occluded"
        ]
        partial_candidates = [
            cattle_id
            for cattle_id, item in visibility.items()
            if item["visibility_status"] == "partially_occluded"
        ]
        base = Image.open(self.current_base_map_path).convert("RGB")
        scene = Image.new("RGB", base.size, (130, 120, 94))
        scene.paste(base, camera_offset)
        self._draw_dynamic_water(scene, camera_offset)
        scene = self._render_cattle(
            scene,
            visible_positions,
            metadata["hora"],
            camera_offset,
            visibility,
        )
        raw = self._apply_camera_effects(
            scene,
            frame_index,
            metadata["hora"],
        )
        raw_path = self.raw_dir / f"frame_{frame_index:04d}.jpg"
        raw.save(
            raw_path,
            format=self.config.get("raw_format", "JPEG"),
            quality=int(self.config.get("jpeg_quality", 90)),
        )

        detections, labels = self._detect(
            visible_positions,
            frame_index,
            camera_offset,
            visibility,
        )
        processed = Image.open(raw_path).convert("RGB")
        processed = self._draw_processed_overlay(
            processed,
            detections,
            metadata,
            camera_offset,
        )
        processed_path = self.output_dir / f"frame_{frame_index:04d}.png"
        processed.save(processed_path)

        label_path = self.labels_dir / f"frame_{frame_index:04d}.txt"
        label_path.write_text("\n".join(labels), encoding="utf-8")
        metadata["render_outputs"] = {
            "raw_drone_frame": str(raw_path),
            "processed_frame": str(processed_path),
            "yolo_labels": str(label_path),
        }
        metadata["camera_offset_px"] = list(camera_offset)
        metadata["detections"] = detections
        metadata["visibility_summary"] = {
            "physical_in_scene": len(cattle_positions),
            "visible_from_drone": len(visible_positions),
            "fully_occluded": len(fully_occluded),
            "partially_occluded": len(partial_candidates),
        }
        metadata["detection_summary"] = {
            "expected_visible": len(visible_positions),
            "detected": len(detections),
            "missed_visible": len(visible_positions) - len(detections),
        }
        self.last_visibility_ground_truth = visibility
        self.last_positions = {
            animal["id"]: (animal["x"], animal["y"])
            for animal in cattle_positions
        }
        return processed_path
