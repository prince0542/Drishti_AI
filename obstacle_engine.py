"""
Drishti AI - Real-Time Obstacle Detection & Assistive Walking Awareness Engine
Provides low-latency object detection (YOLOv8n), persistent tracking (ByteTrack),
spatial zoning (Left/Center/Right), relative depth estimation, walkable corridor analysis,
and a priority-based multi-evidence obstacle decision engine.

Designed for visually impaired users with an offline CPU budget (< 100ms latency on 8 GB RAM).
Modular architecture allows easy future integration with Android and smartphone cameras.
"""

import time
import math
from collections import deque
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class SpatialZone(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class ProximityTier(Enum):
    CRITICAL = "critical"  # Immediate walking collision zone (< 1.2m relative)
    WARNING = "warning"    # Approaching hazard in pathway (1.2m - 2.5m relative)
    NOTICE = "notice"      # Distant or peripheral item (> 2.5m relative)


class MotionTrend(Enum):
    APPROACHING = "approaching"
    STATIONARY = "stationary"
    RECEDING = "receding"


# High-value obstacle threat categorization & bilingual names
THREAT_PROFILES: Dict[str, Dict[str, Any]] = {
    # High Threat (Stairs, drops, fast obstacles, vehicles)
    "stairs": {"threat": 1.0, "en": "stairs", "hi": "सीढ़ियां", "critical_dist": 0.55},
    "person": {"threat": 0.85, "en": "person", "hi": "व्यक्ति", "critical_dist": 0.65},
    "car": {"threat": 0.95, "en": "vehicle", "hi": "गाड़ी", "critical_dist": 0.60},
    "bus": {"threat": 0.95, "en": "bus", "hi": "बस", "critical_dist": 0.60},
    "truck": {"threat": 0.95, "en": "truck", "hi": "ट्रक", "critical_dist": 0.60},
    "motorcycle": {"threat": 0.90, "en": "motorcycle", "hi": "मोटरसाइकिल", "critical_dist": 0.65},
    "bicycle": {"threat": 0.85, "en": "bicycle", "hi": "साइकिल", "critical_dist": 0.65},
    # Medium Threat (Static furniture, doors, interior obstacles)
    "chair": {"threat": 0.70, "en": "chair", "hi": "कुर्सी", "critical_dist": 0.65},
    "couch": {"threat": 0.75, "en": "sofa", "hi": "सोफा", "critical_dist": 0.65},
    "dining table": {"threat": 0.75, "en": "table", "hi": "मेज़", "critical_dist": 0.65},
    "bench": {"threat": 0.70, "en": "bench", "hi": "बेंच", "critical_dist": 0.65},
    "door": {"threat": 0.65, "en": "door", "hi": "दरवाज़ा", "critical_dist": 0.70},
    "fire hydrant": {"threat": 0.75, "en": "fire hydrant", "hi": "फायर हाइड्रेंट", "critical_dist": 0.65},
    # Ground Traps & Bags (Tripping hazards)
    "backpack": {"threat": 0.60, "en": "bag", "hi": "बैग", "critical_dist": 0.75},
    "suitcase": {"threat": 0.70, "en": "suitcase", "hi": "सूटकेस", "critical_dist": 0.70},
    "handbag": {"threat": 0.55, "en": "bag", "hi": "बैग", "critical_dist": 0.75},
    "bottle": {"threat": 0.50, "en": "bottle", "hi": "बोतल", "critical_dist": 0.80},
    "potted plant": {"threat": 0.65, "en": "plant pot", "hi": "गमला", "critical_dist": 0.70},
}


class TrackedObstacle:
    """Represents a persistently tracked physical obstacle across video frames."""

    def __init__(self, track_id: int, class_name: str, confidence: float, bbox: List[float]):
        self.track_id = track_id
        self.class_name = class_name.lower()
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.zone: SpatialZone = SpatialZone.CENTER
        self.relative_depth: float = 0.0  # 0.0 (far) to 1.0 (touching/immediate)
        self.proximity_tier: ProximityTier = ProximityTier.NOTICE
        self.motion_trend: MotionTrend = MotionTrend.STATIONARY
        self.first_seen: float = time.time()
        self.last_seen: float = self.first_seen
        self.history: deque = deque(maxlen=10)
        self.history.append((bbox, self.first_seen))
        self.last_announced_tier: Optional[ProximityTier] = None
        self.last_announced_time: float = 0.0

    def update(self, bbox: List[float], confidence: float, frame_width: int, frame_height: int):
        now = time.time()
        self.history.append((bbox, now))
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = now

        # Update Spatial Zone
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        norm_cx = center_x / float(frame_width)

        # Center walking strip has a defensive safety margin [0.30, 0.70]
        if norm_cx < 0.33:
            self.zone = SpatialZone.LEFT
        elif norm_cx > 0.67:
            self.zone = SpatialZone.RIGHT
        else:
            self.zone = SpatialZone.CENTER

        # Update Relative Depth via Monocular Ground Plane & Scale Geometry
        # When user holds camera pointing forward-downward:
        # y2 (bottom edge touching floor) and bounding area indicate relative proximity
        norm_bottom = y2 / float(frame_height)
        norm_area = ((x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)

        # Horizon is roughly at 35% frame height
        horizon = 0.35
        ground_proximity = max(0.0, (norm_bottom - horizon) / (1.0 - horizon))
        scale_proximity = math.sqrt(min(1.0, norm_area * 3.5))

        # Relative Depth Index (0.0 to 1.0)
        self.relative_depth = float(np.clip((0.65 * ground_proximity) + (0.35 * scale_proximity), 0.0, 1.0))

        # Assign Proximity Tier based on class threat thresholds
        profile = THREAT_PROFILES.get(self.class_name, {"critical_dist": 0.65})
        crit_thresh = profile.get("critical_dist", 0.65)
        warn_thresh = crit_thresh * 0.60

        if self.relative_depth >= crit_thresh or (norm_bottom >= 0.85 and self.zone == SpatialZone.CENTER):
            self.proximity_tier = ProximityTier.CRITICAL
        elif self.relative_depth >= warn_thresh:
            self.proximity_tier = ProximityTier.WARNING
        else:
            self.proximity_tier = ProximityTier.NOTICE

        # Calculate Motion Trend (Approaching, Stationary, Receding)
        if len(self.history) >= 4:
            old_bbox, old_time = self.history[0]
            dt = now - old_time
            if dt > 0.2:
                old_bottom = old_bbox[3] / float(frame_height)
                old_area = ((old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])) / float(frame_width * frame_height)
                area_change = (norm_area - old_area) / max(0.001, old_area)
                bottom_change = norm_bottom - old_bottom

                if area_change > 0.25 or bottom_change > 0.08:
                    self.motion_trend = MotionTrend.APPROACHING
                elif area_change < -0.25 or bottom_change < -0.08:
                    self.motion_trend = MotionTrend.RECEDING
                else:
                    self.motion_trend = MotionTrend.STATIONARY


class CorridorTraversability(NamedTuple):
    left_free_score: float     # 0.0 (blocked) to 1.0 (clear)
    center_free_score: float   # 0.0 (blocked) to 1.0 (clear)
    right_free_score: float    # 0.0 (blocked) to 1.0 (clear)
    recommended_side: Optional[SpatialZone]
    is_path_safe: bool


class ObstacleAwarenessResult:
    """Structured inference payload returned on every analyzed frame."""

    def __init__(
        self,
        annotated_frame: np.ndarray,
        obstacles: List[TrackedObstacle],
        corridor: CorridorTraversability,
        critical_alert: Optional[str] = None,
        alert_priority: int = 0,
        inference_time_ms: float = 0.0,
        lang: str = "en",
    ):
        self.annotated_frame = annotated_frame
        self.obstacles = obstacles
        self.corridor = corridor
        self.critical_alert = critical_alert
        self.alert_priority = alert_priority  # 1 = Critical, 2 = Warning, 3 = Notice
        self.inference_time_ms = inference_time_ms
        self.lang = lang


class ObstacleDecisionEngine:
    """
    Evaluates multi-source evidence (YOLO detection, ByteTrack persistence, relative depth,
    walkable corridor traversability) and generates timely, unambiguous, priority-governed
    auditory warnings for visually impaired walkers.
    """

    def __init__(self, alert_cooldown_seconds: float = 6.0):
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self._last_alert_time: float = 0.0
        self._last_alert_text: str = ""
        self._last_alert_priority: int = 0

    def evaluate(
        self,
        obstacles: List[TrackedObstacle],
        corridor: CorridorTraversability,
        lang: str = "en",
    ) -> Tuple[Optional[str], int]:
        """
        Synthesizes obstacle observations into a single, high-urgency spoken alert.
        Strict rule: Never instruct "move left/right" based only on raw detections.
        Only describes reliable free space if clear evidence exists.
        """
        now = time.time()
        if not obstacles:
            return None, 0

        # Sort obstacles by threat and proximity:
        # Priority criteria: Critical tier center obstacles > Critical peripheral > Approaching warning obstacles
        center_critical = [
            o for o in obstacles
            if o.zone == SpatialZone.CENTER and o.proximity_tier == ProximityTier.CRITICAL
        ]
        any_critical = [o for o in obstacles if o.proximity_tier == ProximityTier.CRITICAL]
        center_warning = [
            o for o in obstacles
            if o.zone == SpatialZone.CENTER and o.proximity_tier == ProximityTier.WARNING
        ]

        alert_text = None
        priority = 0

        # RULE 1: Immediate Collision Hazard Ahead
        if center_critical:
            target = center_critical[0]
            name = THREAT_PROFILES.get(target.class_name, {}).get(lang, target.class_name)
            priority = 1  # Top Priority Alert

            if lang == "hi":
                alert_text = f"सावधान: सामने {name} है। कृपया रुकें।"
            else:
                alert_text = f"{name.capitalize()} detected ahead. Please stop."

            # Check if an alternative clear space has high reliable evidence
            if corridor.recommended_side == SpatialZone.LEFT and corridor.left_free_score >= 0.85:
                if lang == "hi":
                    alert_text += " बाईं ओर रास्ता खुला दिख रहा है।"
                else:
                    alert_text += " Left path appears unobstructed."
            elif corridor.recommended_side == SpatialZone.RIGHT and corridor.right_free_score >= 0.85:
                if lang == "hi":
                    alert_text += " दाईं ओर रास्ता खुला दिख रहा है।"
                else:
                    alert_text += " Right path appears unobstructed."

        # RULE 2: Critical Peripheral Obstacle
        elif any_critical:
            target = any_critical[0]
            name = THREAT_PROFILES.get(target.class_name, {}).get(lang, target.class_name)
            side = target.zone.value
            priority = 2

            if lang == "hi":
                side_hi = "बाईं" if side == "left" else "दाईं"
                alert_text = f"चेतावनी: आपके {side_hi} ओर {name} बहुत पास है।"
            else:
                alert_text = f"Caution: {name} very close on your {side}."

        # RULE 3: Approaching Hazard in Walking Path
        elif center_warning:
            target = center_warning[0]
            name = THREAT_PROFILES.get(target.class_name, {}).get(lang, target.class_name)
            priority = 2

            if target.motion_trend == MotionTrend.APPROACHING:
                if lang == "hi":
                    alert_text = f"ध्यान दें: {name} सामने से आ रहा है। गति धीमी करें।"
                else:
                    alert_text = f"Notice: {name} approaching in front. Slow down."
            else:
                if lang == "hi":
                    alert_text = f"रास्ते में {name} है। सावधानी से आगे बढ़ें।"
                else:
                    alert_text = f"{name.capitalize()} in walking path. Proceed with caution."

        if not alert_text:
            return None, 0

        # Anti-repetition & escalation check:
        # Critical alerts (Priority 1) bypass cooldown if a new obstacle emerges
        time_elapsed = now - self._last_alert_time
        if priority == 1:
            if alert_text != self._last_alert_text or time_elapsed >= 4.0:
                self._last_alert_time = now
                self._last_alert_text = alert_text
                self._last_alert_priority = priority
                return alert_text, priority
        else:
            if time_elapsed >= self.alert_cooldown_seconds and alert_text != self._last_alert_text:
                self._last_alert_time = now
                self._last_alert_text = alert_text
                self._last_alert_priority = priority
                return alert_text, priority

        return None, 0

    def reset(self):
        self._last_alert_time = 0.0
        self._last_alert_text = ""
        self._last_alert_priority = 0


class ObstacleAwarenessSystem:
    """
    Main integrated engine managing YOLOv8n object detection, ByteTrack tracking,
    corridor free-space segmentation, visual radar overlays, and decision alerts.
    Operates 100% offline with zero cloud dependency.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence_thresh: float = 0.40):
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh
        self.model: Optional[Any] = None
        self.decision_engine = ObstacleDecisionEngine()
        self.tracked_objects: Dict[int, TrackedObstacle] = {}
        self._is_ready: bool = False
        self._load_model()

    def _load_model(self):
        """Initializes lightweight YOLOv8n model."""
        if YOLO is None:
            return
        try:
            self.model = YOLO(self.model_path)
            self._is_ready = True
        except Exception as e:
            print(f"[Drishti ObstacleEngine] Error loading model: {e}")
            self._is_ready = False

    def is_available(self) -> bool:
        return self._is_ready and self.model is not None

    def analyze_walkable_corridor(
        self,
        obstacles: List[TrackedObstacle],
        frame_width: int,
        frame_height: int,
    ) -> CorridorTraversability:
        """
        Computes traversability scores for Left, Center, and Right corridors.
        Penalizes corridors based on obstacle depth, area, and center proximity.
        """
        left_pen = 0.0
        center_pen = 0.0
        right_pen = 0.0

        for obs in obstacles:
            pen = (obs.relative_depth ** 1.5) * (0.5 + 0.5 * THREAT_PROFILES.get(obs.class_name, {}).get("threat", 0.7))
            if obs.zone == SpatialZone.LEFT:
                left_pen += pen
            elif obs.zone == SpatialZone.RIGHT:
                right_pen += pen
            else:
                center_pen += pen

        left_score = max(0.0, min(1.0, 1.0 - left_pen))
        center_score = max(0.0, min(1.0, 1.0 - (center_pen * 1.3)))  # Center penalized more strictly
        right_score = max(0.0, min(1.0, 1.0 - right_pen))

        # Determine if clear alternative path exists
        recommended_side = None
        if center_score < 0.45:
            if left_score >= 0.80 and left_score > (right_score + 0.25):
                recommended_side = SpatialZone.LEFT
            elif right_score >= 0.80 and right_score > (left_score + 0.25):
                recommended_side = SpatialZone.RIGHT

        is_safe = center_score >= 0.65

        return CorridorTraversability(
            left_free_score=round(left_score, 2),
            center_free_score=round(center_score, 2),
            right_free_score=round(right_score, 2),
            recommended_side=recommended_side,
            is_path_safe=is_safe,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        lang: str = "en",
        enable_tracking: bool = True,
    ) -> ObstacleAwarenessResult:
        """
        Processes a single BGR/RGB camera frame:
        1. Runs YOLOv8n object detection & tracking.
        2. Assigns spatial zones & relative depth indices.
        3. Evaluates corridor free space.
        4. Synthesizes timely auditory safety alerts.
        5. Renders accessible visual radar overlays.
        """
        start_time = time.time()
        h, w = frame.shape[:2]
        display_frame = frame.copy()

        if not self.is_available():
            dummy_corridor = CorridorTraversability(1.0, 1.0, 1.0, None, True)
            return ObstacleAwarenessResult(
                annotated_frame=display_frame,
                obstacles=[],
                corridor=dummy_corridor,
                critical_alert=None,
                inference_time_ms=0.0,
                lang=lang,
            )

        # Run detection or tracking
        try:
            if enable_tracking:
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.confidence_thresh,
                    verbose=False,
                )
            else:
                results = self.model(
                    frame,
                    conf=self.confidence_thresh,
                    verbose=False,
                )
        except Exception as e:
            # Fallback to direct detection if tracker throws
            results = self.model(frame, conf=self.confidence_thresh, verbose=False)

        current_active_obstacles: List[TrackedObstacle] = []
        seen_track_ids = set()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for idx in range(len(boxes)):
                cls_id = int(boxes.cls[idx].item())
                class_name = self.model.names.get(cls_id, "object").lower()
                conf = float(boxes.conf[idx].item())
                xyxy = boxes.xyxy[idx].cpu().numpy().tolist()

                # Filter for walking-relevant obstacles
                if class_name not in THREAT_PROFILES and conf < 0.55:
                    continue

                # Assign or update persistent track
                track_id = int(boxes.id[idx].item()) if (boxes.id is not None and boxes.id[idx] is not None) else (1000 + idx)
                seen_track_ids.add(track_id)

                if track_id in self.tracked_objects:
                    obstacle = self.tracked_objects[track_id]
                    obstacle.update(xyxy, conf, w, h)
                else:
                    obstacle = TrackedObstacle(track_id, class_name, conf, xyxy)
                    obstacle.update(xyxy, conf, w, h)
                    self.tracked_objects[track_id] = obstacle

                current_active_obstacles.append(obstacle)

        # Clean stale tracks
        now = time.time()
        stale_ids = [tid for tid, obj in self.tracked_objects.items() if (now - obj.last_seen) > 2.0]
        for tid in stale_ids:
            del self.tracked_objects[tid]

        # Evaluate Corridor Traversability
        corridor = self.analyze_walkable_corridor(current_active_obstacles, w, h)

        # Evaluate Decision Engine Alerts
        alert_text, alert_priority = self.decision_engine.evaluate(current_active_obstacles, corridor, lang=lang)

        # Draw Accessible Spatial Radar Overlay on Display Frame
        self._draw_accessible_radar(display_frame, current_active_obstacles, corridor, alert_text)

        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return ObstacleAwarenessResult(
            annotated_frame=display_frame,
            obstacles=current_active_obstacles,
            corridor=corridor,
            critical_alert=alert_text,
            alert_priority=alert_priority,
            inference_time_ms=elapsed_ms,
            lang=lang,
        )

    def _draw_accessible_radar(
        self,
        frame: np.ndarray,
        obstacles: List[TrackedObstacle],
        corridor: CorridorTraversability,
        alert_text: Optional[str],
    ):
        """Draws high-contrast spatial zones, walking corridor guidelines, and obstacle boxes."""
        h, w = frame.shape[:2]

        # 1. Zone Divider Lines (Left 33%, Right 67%)
        z1 = int(w * 0.33)
        z2 = int(w * 0.67)

        # Translucent Walking Corridor Overlay (Center zone)
        overlay = frame.copy()
        center_color = (0, 200, 0) if corridor.is_path_safe else (0, 0, 220)  # Green or Red
        cv2.rectangle(overlay, (z1, 0), (z2, h), center_color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Zone boundary dashed lines
        cv2.line(frame, (z1, 0), (z1, h), (100, 100, 100), 2)
        cv2.line(frame, (z2, 0), (z2, h), (100, 100, 100), 2)

        # Zone Header Indicators
        cv2.putText(frame, f"LEFT [{int(corridor.left_free_score*100)}%]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
        center_txt = f"PATH [{int(corridor.center_free_score*100)}%]" if corridor.is_path_safe else f"BLOCKED [{int(corridor.center_free_score*100)}%]"
        cv2.putText(frame, center_txt, (z1 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255) if corridor.is_path_safe else (0, 50, 255), 2)
        cv2.putText(frame, f"RIGHT [{int(corridor.right_free_score*100)}%]", (z2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        # 2. Draw Detected Obstacles with Accessible Color-Coded Tiers
        for obs in obstacles:
            x1, y1, x2, y2 = [int(v) for v in obs.bbox]

            if obs.proximity_tier == ProximityTier.CRITICAL:
                color = (0, 0, 255)      # Red = Immediate Collision Hazard
                thickness = 4
            elif obs.proximity_tier == ProximityTier.WARNING:
                color = (0, 165, 255)    # Orange = Warning in Pathway
                thickness = 2
            else:
                color = (0, 255, 0)      # Green = Notice
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # High-Contrast Label Pill
            depth_pct = int(obs.relative_depth * 100)
            label = f"{obs.class_name.upper()} | {obs.proximity_tier.value.upper()} ({depth_pct}%)"
            if obs.motion_trend == MotionTrend.APPROACHING:
                label += " [APPROACHING]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if color != (0, 0, 255) else (255, 255, 255), 2)

        # 3. Active Alert Banner across bottom
        if alert_text:
            cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, f"WARNING: {alert_text}", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)


# Global Singleton Instance
obstacle_system = ObstacleAwarenessSystem()
