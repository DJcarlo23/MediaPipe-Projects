import cv2
import numpy as np
from HandTrackingModule import handDetector

cap = cv2.VideoCapture(0)
detector = handDetector()

count_true = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        
        thumb = lmList[4][1] > lmList[5][1]
        index = lmList[8][2] < lmList[6][2]
        middle = lmList[12][2] < lmList[10][2]
        ring = lmList[16][2] < lmList[14][2]
        little = lmList[20][2] < lmList[18][2]

        count_list = np.array([thumb, index, middle, ring, little])
        count_true = np.count_nonzero(count_list)

    cv2.putText(img, "{}".format(count_true), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)