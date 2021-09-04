import cv2
import numpy as np
import pickle
from HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector()

# load the model from disk
filename = 'HandsTracking/Hand Language Signs Recognizer/hand_language_prediction_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # lmArray = np.array(lmList).reshape(1, -1)

    if len(lmList) != 0 and len(lmList) == 63:
        lmArray = np.array(lmList).reshape(1, -1)
        result = loaded_model.predict(lmArray)
        print(result)
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)