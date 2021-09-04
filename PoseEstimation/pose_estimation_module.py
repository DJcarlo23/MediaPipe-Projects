import cv2
import mediapipe as mp
import time

class PoseEstimation():
    """Class provides tools to work with mediapipe pose estimation

    Attributes:
    min_def_conf : float
        minimum detection confidence
    min_track_conf : float
        minimum tracking confidance
    
    Method:
    find_landmarks : list
        return landmarks position
    pose_estimation : cv2 image
        return image with the landmarks
    """
    def __init__(self, min_det_conf=0.5, min_track_conf=0.5):
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.mp_pose_fun = self.mp_pose.Pose(
            min_detection_confidence = self.min_det_conf,
            min_tracking_confidence= self.min_track_conf
        )

    def find_landmarks(self, img):
        """Return list of the landmarks on the given image

        Keyword arguments:
        img -- the image of the person

        Return:
        lmList -- list of the landmarks
        
        """
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_pose_fun.process(image)

        lmList = results.pose_landmarks

        return lmList

    def pose_estimation(self, img):
        """Return person image with the landmarks on it

        Keyword arguments:
        img -- the image of the face

        Return:
        image -- image of the face with mesh
        
        """
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_pose_fun.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        return image

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseEstimation()

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        # lmList = detector.find_landmarks(img)
        image = detector.pose_estimation(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("MediaPipe Pose", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == "__main__":
    main()