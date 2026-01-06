import mediapipe as mp
import cv2
from keras.models import load_model
import logichelper
import numpy as np

right_prediction = [0,0,0]
left_prediction = [0,0,0]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def decimal_to_percent(decimal):
    percent = decimal*100
    percent = round(percent)
    return str(percent) + '%'

def apply_prediction_text(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    leftoffset = 20
    rightoffset = 1000

    cv2.putText(image, 'LEFT HAND', (leftoffset,50), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'ROCK:     ' + decimal_to_percent(left_prediction[0]), (leftoffset,100), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'PAPER:    ' + decimal_to_percent(left_prediction[1]), (leftoffset,150), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'SCISSORS: ' + decimal_to_percent(left_prediction[2]), (leftoffset,200), font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.putText(image, 'RIGHT HAND', (rightoffset,50), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'ROCK:     ' + decimal_to_percent(right_prediction[0]), (rightoffset,100), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'PAPER:    ' + decimal_to_percent(right_prediction[1]), (rightoffset,150), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, 'SCISSORS: ' + decimal_to_percent(right_prediction[2]), (rightoffset,200), font, fontScale, color, thickness, cv2.LINE_AA)

def apply_game_result(image, gameresult, lhl, rhl):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    winning_color = (255, 0, 0)
    losing_color = (0, 0, 255)
    draw_color = (0, 255, 0)

    left_color = draw_color
    right_color = draw_color

    left_text = 'DRAW'
    right_text = 'DRAW'

    thickness = 5

    if gameresult == 0:
        left_color = winning_color
        right_color = losing_color
        left_text = 'WINNER'
        right_text = 'LOSER'
    elif gameresult == 1:
        left_color = losing_color
        right_color = winning_color
        left_text = 'LOSER'
        right_text = 'WINNER'


    rx = [landmark.x for landmark in lhl.landmark]
    ry = [landmark.y for landmark in lhl.landmark]

    lx = [landmark.x for landmark in rhl.landmark]
    ly = [landmark.y for landmark in rhl.landmark]

    height, width, channels = image.shape

    left_center = np.array([np.mean(rx)*width, np.mean(ry)*height]).astype('int32')
    right_center = np.array([np.mean(lx)*width, np.mean(ly)*height]).astype('int32')

    boxoffset = 150

    cv2.putText(image, left_text, (left_center[0]-boxoffset,left_center[1]-boxoffset-25), font, fontScale, left_color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (left_center[0]-boxoffset,left_center[1]-boxoffset), (left_center[0]+boxoffset,left_center[1]+boxoffset), left_color, thickness)
    cv2.putText(image, right_text, (right_center[0]-boxoffset,right_center[1]-boxoffset-25), font, fontScale, right_color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (right_center[0]-boxoffset,right_center[1]-boxoffset), (right_center[0]+boxoffset,right_center[1]+boxoffset), right_color, thickness)

model = load_model('models/rock-paper-scissors.h5')

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

        if lhl != None:
            relevantlandmarks = logichelper.relevant_landmark_vector(lhl.landmark)
            result = model.predict([relevantlandmarks], batch_size=1, verbose=0)
            right_prediction = result[0]
        else:
            right_prediction = [0,0,0]


        rhl = results.right_hand_landmarks
        mp_drawing.draw_landmarks(image, rhl, mp_holistic.HAND_CONNECTIONS)

        if rhl != None:
            relevantlandmarks = logichelper.relevant_landmark_vector(rhl.landmark)
            result = model.predict([relevantlandmarks], batch_size=1, verbose=0)
            left_prediction = result[0]
        else:
            left_prediction = [0,0,0]


        if lhl != None and rhl != None:
            gameresult = logichelper.determine_game_winner(left_prediction, right_prediction)
            apply_game_result(image, gameresult, lhl, rhl)



        apply_prediction_text(image)
        cv2.imshow('Live Inference' , image)

        keyPress = cv2.waitKey(50)

        if keyPress == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()
