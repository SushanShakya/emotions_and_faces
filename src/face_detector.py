import os
import sys
import cv2


class FaceDetector:
    def __init__(self, model_path=None, asset_path=None):
        self.model_path = model_path if model_path is not None else self.default_path()
        self.asset_path = (
            asset_path if asset_path is not None else self.default_asset_path()
        )
        self.classifier = cv2.CascadeClassifier(self.model_path)

    def base_dir(self):
        return os.path.dirname(os.path.abspath(sys.argv[0]))

    def get_abs_path(self, path):
        return self.base_dir() + path

    def default_asset_path(self):
        return self.get_abs_path("/assets/me.jpg")

    def default_path(self):
        cvdir = os.path.dirname(cv2.__file__)
        xmlpath = "/data/haarcascade_frontalface_default.xml"
        return cvdir + xmlpath

    def __detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    def output_with_rectangles(self):
        img = cv2.imread(self.asset_path)
        faces = self.__detect(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imwrite(self.get_abs_path("/assets/out.jpg"), img)
        cv2.destroyAllWindows()

    def detect(self):
        img = cv2.imread(self.asset_path)
        faces = self.__detect(img)
        cropped_faces = [img[y : y + h, x : x + w] for (x, y, w, h) in faces]
        return cropped_faces
