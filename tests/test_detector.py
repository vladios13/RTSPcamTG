import time

import numpy as np

from app import detector


def test_load_classes_populates_classes_and_colors():
    detector.classes = None
    detector.COLORS = None
    detector.load_classes()
    assert detector.classes is not None
    assert len(detector.classes) == 80
    assert detector.classes[0] == 'person'
    assert detector.COLORS.shape == (80, 3)


class TestLetterbox:
    def test_square_output(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        padded, ratio, (pad_left, pad_top) = detector.letterbox(img, detector.INPUT_SIZE)
        assert padded.shape[:2] == (detector.INPUT_SIZE, detector.INPUT_SIZE)

    def test_wide_frame_pads_vertically_only(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 16:9
        padded, ratio, (pad_left, pad_top) = detector.letterbox(img, detector.INPUT_SIZE)
        assert pad_left == 0
        assert pad_top > 0

    def test_tall_frame_pads_horizontally_only(self):
        img = np.zeros((1280, 720, 3), dtype=np.uint8)  # 9:16
        padded, ratio, (pad_left, pad_top) = detector.letterbox(img, detector.INPUT_SIZE)
        assert pad_top == 0
        assert pad_left > 0


class TestGetImageDifference:
    def test_identical_frames_are_zero(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert detector.get_image_difference(img, img.copy()) == 0.0

    def test_different_shapes_are_one(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.zeros((50, 50, 3), dtype=np.uint8)
        assert detector.get_image_difference(a, b) == 1.0

    def test_different_frames_are_between_zero_and_one(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.full((100, 100, 3), 255, dtype=np.uint8)
        diff = detector.get_image_difference(a, b)
        assert 0.0 < diff <= 1.0


class TestCheckAlarm:
    def setup_method(self):
        detector.notified = []

    def test_same_object_same_spot_is_suppressed_within_window(self):
        assert detector.checkAlarm('cam1', 'person', (100, 100)) is True
        assert detector.checkAlarm('cam1', 'person', (100, 100)) is False

    def test_different_camera_is_not_suppressed(self):
        assert detector.checkAlarm('cam1', 'person', (100, 100)) is True
        assert detector.checkAlarm('cam2', 'person', (100, 100)) is True

    def test_far_enough_point_is_not_suppressed(self):
        assert detector.checkAlarm('cam1', 'person', (100, 100)) is True
        assert detector.checkAlarm('cam1', 'person', (200, 200)) is True

    def test_outside_window_is_not_suppressed(self):
        now = time.time()
        detector.notified = [
            {'cam': 'cam1', 'name': 'person', 'point': (100, 100), 'time': now - 301},
        ]
        assert detector.checkAlarm('cam1', 'person', (100, 100)) is True
