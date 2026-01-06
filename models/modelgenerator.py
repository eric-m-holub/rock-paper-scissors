import keras
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras import regularizers

import json
import numpy as np
import os


def generateModel(newData):

    if newData != None:
        collectedData = newData
    else:
        with open('../data/rock-paper-scissors.json') as json_file:
            json_data = json.load(json_file)
            collectedData = json_data

    train_samples = []
    train_labels = []

    for data in collectedData:
        train_samples.append(data['relevantlandmarks'])
        train_labels.append(data['label'])

    dimensions = 0
    if len(train_samples) > 0:
        dimensions = len(train_samples[0])



    if os.path.exists('models/rock-paper-scissors.h5'):
        model = load_model('models/rock-paper-scissors.h5')
    else:
        model = Sequential([
            Dense(16, input_shape=(dimensions,), activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])

    model.compile(Adam(lr=.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(x=train_samples, y=train_labels, batch_size=10, epochs=200, shuffle=True, verbose=2)

    if newData != None:
        model.save('models/rock-paper-scissors.h5')
    else:
        model.save('rock-paper-scissors.h5')
