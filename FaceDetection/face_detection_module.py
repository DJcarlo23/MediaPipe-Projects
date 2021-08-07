import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_detection

class FaceDetector():
    def __init__(self, model=0, detection_conf=0.5):
        self.model = model
        self.detection_conf = detection_conf

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

    def find_head_position(self, img):
        face_detection = self.mp_face_detection.FaceDetection(
            model_selection = self.model,
            min_detection_confidence = self.detection_conf
        )

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = face_detection.process(image)
        
        return self.results.detections

    def find_head(self, img):
        # image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.results.detections:
            for detection in self.results.detections:
                self.mp_drawing.draw_detection(img, detection)

        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        sucess, img = cap.read()
        lmList = detector.find_head_position(img)
        image = detector.find_head(img)

        cv2.imshow("Image", image)
        cv2.waitKey(1)

        print(lmList)

if __name__ == "__main__":
    main()