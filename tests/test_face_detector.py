from unittest import TestCase

from src.face_detector import FaceDetector


class TestFaceDetector(TestCase):
    def test_create(self):
        model_path = ""
        FaceDetector(model_path)

    def test_detect(self):
        f = FaceDetector()
        f.detect_from_webcam()
