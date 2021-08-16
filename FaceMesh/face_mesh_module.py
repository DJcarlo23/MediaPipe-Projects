import cv2
import mediapipe as mp

class FaceMesh():
    def __init__(self, min_det_conf=0.5, min_track_conf=0.5, max_num_faces=1):
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf
        self.max_num_faces = max_num_faces



def main():
    print("Works")

if __name__ == "__main__":
    main()