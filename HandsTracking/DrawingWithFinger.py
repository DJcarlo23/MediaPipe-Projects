import cv2
import time
import os
from HandTrackingModule import handDetector

cap = cv2.VideoCapture(0)
detector = handDetector()

to_draw = []
colors = [
    (255, 0, 255),
    (0, 0, 255),
    (255, 0, 0)
]

counter = 0
flag = True
current_color = colors[counter]
    

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0:

        # use your fist to clear image
        if lmList[8][2] > lmList[6][2]:
            to_draw = []
        else:
            to_draw.append([lmList[8][1], lmList[8][2]])

            for point in to_draw:
                cv2.circle(img, (point[0], point[1]), 5, current_color, cv2.FILLED)

        # change color with second finger
        if lmList[12][2] < lmList[10][2]:
            if flag:
                counter += 1

                if counter >= 3:
                    counter = 0
                
                current_color = colors[counter]
                flag = False
            
        if lmList[12][2] > lmList[10][2]:
            flag = True



    cv2.imshow("Image", img)
    cv2.waitKey(1)