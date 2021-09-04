import cv2
import mediapipe as mp
import time

class FaceMesh():
    """Class provides tools to work with mediapipe face mesh

    Attributes:
    min_det_conf : float
        minimum detection confidence
    min_track_conf : float
        minimum tracking confidance
    max_num_faces : int
        maximum number of faces which our program will be detecting

    Methods:
    find_landmarks : list
        return landmarks position 
    face_mesh : cv2 image
        return image with the landmarks
    """
    def __init__(self, min_det_conf=0.5, min_track_conf=0.5, max_num_faces=1):
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf
        self.max_num_faces = max_num_faces

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh_fun = self.mp_face_mesh.FaceMesh(
            min_detection_confidence = self.min_det_conf,
            min_tracking_confidence = self.min_track_conf,
            max_num_faces = self.max_num_faces
        )

    def find_landmarks(self, img):
        """Return list of the landmarks on the given image

        Keyword arguments:
        img -- the image of the face

        Return:
        lmList -- list of the landmarks
        
        """
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh_fun.process(image)
        lmList = results.multi_face_landmarks

        return lmList
    
    def face_mesh(self, img):
        """Return face image with the mesh on it

        Keyword arguments:
        img -- the image of the face

        Return:
        image -- image of the face with mesh
        
        """
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh_fun.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = self.mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec)

        return image

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMesh()

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        # lmList = detector.find_landmarks(img)
        image = detector.face_mesh(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("MediaPipe FaceMesh", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == "__main__":
    main()