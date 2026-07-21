import unittest

import cv2
import numpy as np

from guidance.path_follower import (
    GuidanceConfig,
    StreetViewReferenceTarget,
    VisualPathFollower,
)


def synthetic_path(shift_px=0):
    frame = np.full((720, 1280, 3), 65, dtype=np.uint8)
    cv2.line(frame, (170 + shift_px, 700), (555 + shift_px, 300), (255, 255, 255), 12)
    cv2.line(frame, (1110 + shift_px, 700), (725 + shift_px, 300), (255, 255, 255), 12)
    return frame


class VisualPathFollowerTest(unittest.TestCase):
    def setUp(self):
        config = GuidanceConfig(command_smoothing=1.0, min_confidence=0.1)
        self.follower = VisualPathFollower(config)

    def test_centred_path_goes_straight(self):
        result = self.follower.process(synthetic_path())
        self.assertGreater(result.confidence, 0.5)
        self.assertAlmostEqual(result.angular_z, 0.0, delta=0.08)

    def test_path_on_right_requests_right_turn(self):
        result = self.follower.process(synthetic_path(shift_px=120))
        self.assertLess(result.angular_z, -0.05)
        self.assertEqual(result.instruction, "TURN RIGHT")

    def test_path_on_left_requests_left_turn(self):
        result = self.follower.process(synthetic_path(shift_px=-120))
        self.assertGreater(result.angular_z, 0.05)
        self.assertEqual(result.instruction, "TURN LEFT")

    def test_empty_image_is_rejected(self):
        with self.assertRaises(ValueError):
            self.follower.process(np.empty((0, 0, 3), dtype=np.uint8))

    def test_streetview_route_turn_changes_command(self):
        target = StreetViewReferenceTarget(
            filename="streetview_0010.jpg",
            match_index=10,
            latitude=50.1,
            longitude=14.3,
            heading_deg=10.0,
            future_heading_deg=55.0,
            vpr_score=80.0,
            expected_lateral_error=0.0,
            expected_heading_error=0.0,
            route_turn_error=0.5,
            confidence=1.0,
        )
        result = self.follower.process(
            synthetic_path(),
            reference_target=target,
        )
        self.assertEqual(result.control_source, "STREETVIEW_DB")
        self.assertEqual(result.reference_match_index, 10)
        self.assertLess(result.angular_z, -0.10)

    def test_required_streetview_match_stops_when_missing(self):
        follower = VisualPathFollower(
            GuidanceConfig(
                command_smoothing=1.0,
                min_confidence=0.1,
                require_streetview_reference=True,
            )
        )
        result = follower.process(synthetic_path())
        self.assertEqual(result.linear_x, 0.0)
        self.assertEqual(result.angular_z, 0.0)
        self.assertEqual(result.instruction, "STOP - NO STREETVIEW MATCH")


if __name__ == "__main__":
    unittest.main()
