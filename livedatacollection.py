import mediapipe as mp
import numpy as np
import cv2
import json
import sys

sys.path.insert(0, 'models')

import modelgenerator
import logichelper

active_training_label = -1

collectedData = []


def apply_active_training_label(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    label = logichelper.index_to_game_choice(active_training_label)
    image = cv2.putText(image, label, org, font, fontScale, color, thickness, cv2.LINE_AA)

def apply_hand_box(image,landmarks,left):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 255, 0)
    if left == False:
        color = color = (0, 0, 255)
    thickness = 2
    boxoffset = 200
    height, width, channels = image.shape

    x = [landmark.x for landmark in landmarks]
    y = [landmark.y for landmark in landmarks]

    center = np.array([np.mean(x)*width, np.mean(y)*height]).astype('int32')
    text = 'TRAINING ' + logichelper.index_to_game_choice(active_training_label) + '...'

    cv2.putText(image, text, (center[0]-boxoffset,center[1]-boxoffset-25), font, fontScale, color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (center[0]-boxoffset,center[1]-boxoffset), (center[0]+boxoffset,center[1]+boxoffset), color, thickness)


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        #recolor feed
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        #make detections
        results = holistic.process(image)

        #recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw right hand features
        lhl = results.left_hand_landmarks
        mp_drawing.draw_landmarks(image, lhl, mp_holistic.HAND_CONNECTIONS)

        if lhl != None and active_training_label > -1:
            newData = {}
            newData['label'] = active_training_label

            relevantlandmarks = logichelper.relevant_landmark_vector(lhl.landmark)
            newData['relevantlandmarks'] = relevantlandmarks

            collectedData.append(newData)
            apply_hand_box(image, lhl.landmark, False)



        rhl = results.right_hand_landmarks
        mp_drawing.draw_landmarks(image, rhl, mp_holistic.HAND_CONNECTIONS)

        if rhl != None and active_training_label > -1:
            newData = {}
            newData['label'] = active_training_label

            relevantlandmarks = logichelper.relevant_landmark_vector(rhl.landmark)
            newData['relevantlandmarks'] = relevantlandmarks

            collectedData.append(newData)
            apply_hand_box(image, rhl.landmark, True)


        apply_active_training_label(image)
        cv2.imshow('Recognition Training' , image)

        keyPress = cv2.waitKey(50)

        if keyPress == ord('q'):
            break
        elif keyPress == ord('1'):
            active_training_label = 0
        elif keyPress == ord('2'):
            active_training_label = 1
        elif keyPress == ord('3'):
            active_training_label = 2
        elif keyPress == ord('0'):
            active_training_label = -1




cap.release()
cv2.destroyAllWindows()


with open('data/rock-paper-scissors.json') as json_file:
    json_data = json.load(json_file)
    with open('data/rock-paper-scissors.json', 'w') as outfile:
        newData = json_data+collectedData
        json.dump(newData, outfile)
        modelgenerator.generateModel(newData)
