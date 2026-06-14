from typing import List, Dict, Any


def _box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(box1: List[float], box2: List[float]) -> float:
    inter = _intersection(box1, box2)
    union = _box_area(box1) + _box_area(box2) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def _is_inside(inner: List[float], outer: List[float]) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _center(box: List[float]) -> (float, float):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _adjacent(box1: List[float], box2: List[float], tolerance: float = 0.04) -> bool:
    if _iou(box1, box2) > 0.0:
        return False
    if abs(box1[2] - box2[0]) < tolerance or abs(box2[2] - box1[0]) < tolerance:
        return True
    if abs(box1[3] - box2[1]) < tolerance or abs(box2[3] - box1[1]) < tolerance:
        return True
    return False


def build_scene_graph(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relations = []
    seen = set()

    for i, source in enumerate(elements):
        for j, target in enumerate(elements):
            if i == j:
                continue
            source_box = source.get("bbox", [0.0, 0.0, 0.0, 0.0])
            target_box = target.get("bbox", [0.0, 0.0, 0.0, 0.0])
            if _is_inside(source_box, target_box) and source_box != target_box:
                relation = (i, "inside", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "inside", "target": j})
                continue
            if _intersection(source_box, target_box) > 0 and _iou(source_box, target_box) > 0.05:
                relation = (i, "overlaps", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "overlaps", "target": j})
                continue
            if source_box[3] <= target_box[1]:
                relation = (i, "above", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "above", "target": j})
            elif source_box[1] >= target_box[3]:
                relation = (i, "below", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "below", "target": j})
            if source_box[2] <= target_box[0]:
                relation = (i, "left_of", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "left_of", "target": j})
            elif source_box[0] >= target_box[2]:
                relation = (i, "right_of", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "right_of", "target": j})
            if _adjacent(source_box, target_box):
                relation = (i, "adjacent_to", j)
                if relation not in seen:
                    seen.add(relation)
                    relations.append({"source": i, "relation": "adjacent_to", "target": j})

    for element in elements:
        bbox = element.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if bbox[1] < 0.12 and bbox[2] - bbox[0] > 0.3:
            relation = (element["id"], "inside", "top_toolbar")
            if relation not in seen:
                seen.add(relation)
                relations.append({"source": element["id"], "relation": "inside", "target": "top_toolbar"})
        if bbox[0] < 0.15 and bbox[3] - bbox[1] > 0.3:
            relation = (element["id"], "inside", "left_sidebar")
            if relation not in seen:
                seen.add(relation)
                relations.append({"source": element["id"], "relation": "inside", "target": "left_sidebar"})
        if bbox[2] > 0.85 and bbox[3] - bbox[1] > 0.3:
            relation = (element["id"], "inside", "right_sidebar")
            if relation not in seen:
                seen.add(relation)
                relations.append({"source": element["id"], "relation": "inside", "target": "right_sidebar"})

    return relations
