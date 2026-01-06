from math import sqrt


def distance_between_vectors(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


def landmark_to_vector(landmark):
    return [landmark.x, landmark.y, landmark.z]


def palm_distance_vector(landmarks):
    ret = []

    palm = landmark_to_vector(landmarks[0])
    landmarksToTarget = [4,8,12,16,20]

    for i in landmarksToTarget:
        landmarkvector = landmark_to_vector(landmarks[i])
        palmdistance = distance_between_vectors(palm,landmarkvector)
        ret.append(palmdistance)

    return ret

def relevant_landmark_vector(landmarks):
    ret = []

    landmarksToTarget = [0,4,8,12,16,20]

    for i in landmarksToTarget:
        landmarkvector = landmark_to_vector(landmarks[i])
        ret = ret + landmarkvector

    return ret


def max_index_of_prediction(prediction):
    return prediction.argmax()


def index_to_game_choice(index):
    if index == 0:
        return 'ROCK'
    elif index == 1:
        return 'PAPER'
    elif index == 2:
        return 'SCISSORS'

    return ''



def determine_game_winner(left_prediction, right_prediction):
    # 0 = rock, 1 = paper, 2 = scissors
    left_choice = max_index_of_prediction(left_prediction)
    right_choice = max_index_of_prediction(right_prediction)

    # 0 = left wins, 1 = right wins, -1 = draw
    left_result = [[-1, 0, 1],[1, -1, 0],[0, 1, -1]]

    return left_result[left_choice][right_choice]
