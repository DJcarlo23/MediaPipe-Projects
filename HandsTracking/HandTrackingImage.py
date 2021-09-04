import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = []

# Find images and put them into IMAGE_FILES list
path = 'D:/Data/Hand Language Signs/asl_alphabet_train/asl_alphabet_train/K'
files = os.listdir(path)
for f in files:
    IMAGE_FILES.append(path + '/' + f)

with mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.7 # default 0.5
) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output
        image = cv2.flip(cv2.imread(file), 1)

        # Convert the BGR image to RGB before processing
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # print("Handedness:", results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )
            
            # Save new file in given place
            cv2.imwrite('D:/Data\Hand Language Signs/Signs tracked by MediaPipe/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))