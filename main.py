from src.emotion_classifier import EmotionClassifier
from src.face_detector import FaceDetector


def main():
    f = FaceDetector()
    c = EmotionClassifier()

    f.output_with_rectangles()

    faces = f.detect()

    for face in faces:
        emotion = c.classify(face)
        print(emotion)


if __name__ == "__main__":
    main()
