import csv
import cv2
import os.path
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def load_data(data_folder):
    lines = []
    with open('{}/driving_log.csv'.format(data_folder)) as csvfile:
        for line in csv.reader(csvfile):
            lines.append(line)
    
    images = []
    measurements = []
    for line in lines:
        path = line[0]
        filename = path.split('/')[-1]
        current_path = '{}/IMG/{}'.format(data_folder,filename)
        if not os.path.isfile(current_path):
            continue
        image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


def test_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model
    
def lenet5_model(input_shape):
    model = Sequential()
    model.add(Lambda (lambda x: x/255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

# data_path = './data'
data_path = '../train'
X_train, y_train = load_data(data_path)
# model = test_model()
model = lenet5_model((160,320,3))
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10) 
model.save('model.h5')

