import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_detection

class FaceDetector():
    """Class provides tools to work with mediapipe face detection
    
    Attributes:
    model : int
        0 is better if we want to detect face close and 1 is better if face is far away
    detection_conf : float
        detection confidence

    Methods:
    find_head_position : list
        returns position of the head on the image
    find_head : cv2 image
        returns a image with the head marked
    """
    def __init__(self, model=0, detection_conf=0.5):
        self.model = model
        self.detection_conf = detection_conf

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

    def find_head_position(self, img):
        """Return position of the head on the image

        Keyword arguments:
        img -- the image of the face

        Return:
        lmList -- list with the position of the head
        """
        face_detection = self.mp_face_detection.FaceDetection(
            model_selection = self.model,
            min_detection_confidence = self.detection_conf
        )

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = face_detection.process(image)
        lmList = self.results
        
        return lmList

    def find_head(self, img):
        """Return a image with the head marked

        Keyword arguments:
        img -- the image of the face

        Return:
        img -- image of the marked head
        
        """
        # image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.results.detections:
            for detection in self.results.detections:
                self.mp_drawing.draw_detection(img, detection)

        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        lmList = detector.find_head_position(img)
        image = detector.find_head(img)

        cv2.imshow("Image", image)
        cv2.waitKey(1)

        print(lmList)

if __name__ == "__main__":
    main()