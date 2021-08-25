import cv2
import mediapipe as mp
import os
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

folders_names = os.listdir('D:/Data/Hand Language Signs/asl_alphabet_train/asl_alphabet_train/')

for folder_name in folders_names:
    IMAGE_FILES = []
    print("I am working with " + folder_name)

    path = 'D:/Data/Hand Language Signs/asl_alphabet_train/asl_alphabet_train/' + folder_name
    files = os.listdir(path)
    for f in files:
        IMAGE_FILES.append(path + '/' + f)

    dataframe_data = []
    idx = 0

    with mp_hands.Hands(
        static_image_mode = True,
        max_num_hands = 2,
        min_detection_confidence = 0.5
    ) as hands:
        for file in IMAGE_FILES:
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
                # print('hand_landmarks:', hand_landmarks)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )

                landmarks_to_df = []

                for landmark in hand_landmarks.landmark:
                    landmarks_to_df.append(landmark.x)
                    landmarks_to_df.append(landmark.y)
                    landmarks_to_df.append(landmark.z)

                dataframe_data.append(landmarks_to_df)
                
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                cv2.imwrite('D:/Data/Hand Language Signs/Signs tracked by MediaPipe/' + folder_name + '/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
                idx += 1
        
        df = pd.DataFrame.from_records(dataframe_data)
        df['Sign name'] = folder_name

        df.to_csv('D:/Data/Hand Language Signs/Signs tracked by MediaPipe/' + folder_name + '/DataFrame/dataframe.csv')