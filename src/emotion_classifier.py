from deepface import DeepFace


class EmotionClassifier:
    def classify(self, img):
        result = DeepFace.analyze(img, actions=["emotion"])
        return result[0]["dominant_emotion"]
