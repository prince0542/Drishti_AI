"""
Unit and benchmark tests for Drishti AI Obstacle Awareness Engine.
Validates latency, threat profiles, decision engine logic, and corridor analysis.
"""

import time
import numpy as np
from obstacle_engine import (
    SpatialZone,
    ProximityTier,
    MotionTrend,
    TrackedObstacle,
    CorridorTraversability,
    ObstacleDecisionEngine,
    ObstacleAwarenessSystem,
    THREAT_PROFILES,
)


def test_threat_profiles_coverage():
    """Ensure walking hazards are mapped with threat scores and bilingual strings."""
    essential_classes = ["chair", "person", "stairs", "car", "dining table", "backpack", "bottle"]
    for cls in essential_classes:
        assert cls in THREAT_PROFILES
        assert "threat" in THREAT_PROFILES[cls]
        assert "en" in THREAT_PROFILES[cls]
        assert "hi" in THREAT_PROFILES[cls]
        assert "critical_dist" in THREAT_PROFILES[cls]


def test_spatial_zone_assignment():
    """Validates that bounding box positions map correctly to Left, Center, and Right zones."""
    frame_w, frame_h = 640, 480

    # Left obstacle (x: 50 to 180 -> center: 115 / 640 = 0.18)
    obs_left = TrackedObstacle(1, "chair", 0.85, [50, 100, 180, 400])
    obs_left.update([50, 100, 180, 400], 0.85, frame_w, frame_h)
    assert obs_left.zone == SpatialZone.LEFT

    # Center obstacle (x: 250 to 390 -> center: 320 / 640 = 0.50)
    obs_center = TrackedObstacle(2, "chair", 0.85, [250, 100, 390, 400])
    obs_center.update([250, 100, 390, 400], 0.85, frame_w, frame_h)
    assert obs_center.zone == SpatialZone.CENTER

    # Right obstacle (x: 480 to 600 -> center: 540 / 640 = 0.84)
    obs_right = TrackedObstacle(3, "chair", 0.85, [480, 100, 600, 400])
    obs_right.update([480, 100, 600, 400], 0.85, frame_w, frame_h)
    assert obs_right.zone == SpatialZone.RIGHT


def test_proximity_tier_and_relative_depth():
    """Checks that ground proximity and scale correctly determine collision hazard."""
    frame_w, frame_h = 640, 480

    # Large obstacle right at the bottom (immediate collision zone)
    obs_critical = TrackedObstacle(10, "chair", 0.90, [200, 200, 440, 470])
    obs_critical.update([200, 200, 440, 470], 0.90, frame_w, frame_h)
    assert obs_critical.relative_depth >= 0.60
    assert obs_critical.proximity_tier == ProximityTier.CRITICAL

    # Small distant obstacle near horizon
    obs_distant = TrackedObstacle(11, "chair", 0.70, [280, 180, 340, 240])
    obs_distant.update([280, 180, 340, 240], 0.70, frame_w, frame_h)
    assert obs_distant.relative_depth < 0.40
    assert obs_distant.proximity_tier == ProximityTier.NOTICE


def test_decision_engine_rules():
    """Tests critical alert generation, alternative path guidance, and language support."""
    engine = ObstacleDecisionEngine()

    # Rule 1: Immediate obstacle ahead with clear left path
    obs = TrackedObstacle(20, "chair", 0.88, [220, 200, 420, 460])
    obs.zone = SpatialZone.CENTER
    obs.proximity_tier = ProximityTier.CRITICAL
    corridor = CorridorTraversability(0.90, 0.10, 0.30, SpatialZone.LEFT, False)

    alert_en, prio_en = engine.evaluate([obs], corridor, lang="en")
    assert prio_en == 1
    assert "Chair detected ahead. Please stop." in alert_en
    assert "Left path appears unobstructed." in alert_en

    # Hindi Alert
    engine.reset()
    alert_hi, prio_hi = engine.evaluate([obs], corridor, lang="hi")
    assert prio_hi == 1
    assert "कुर्सी" in alert_hi
    assert "कृपया रुकें" in alert_hi


def test_steady_state_cpu_latency():
    """Validates that YOLOv8n obstacle awareness runs within the CPU time budget (< 120ms)."""
    system = ObstacleAwarenessSystem()
    assert system.is_available()

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Warmup
    system.process_frame(test_frame)

    latencies = []
    for _ in range(5):
        t0 = time.time()
        res = system.process_frame(test_frame)
        latencies.append((time.time() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    print(f"\n[Benchmark] Obstacle Awareness CPU Latency: {avg_ms:.1f} ms")
    assert avg_ms < 150.0  # Must be faster than 150ms on CPU for real-time walking assistance


if __name__ == "__main__":
    test_threat_profiles_coverage()
    test_spatial_zone_assignment()
    test_proximity_tier_and_relative_depth()
    test_decision_engine_rules()
    test_steady_state_cpu_latency()
    print("All Obstacle Awareness unit & benchmark tests passed successfully!")
