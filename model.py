import csv
import cv2
import os.path 
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#don't use load_data as it loads all images in memory at once, use generators instead
def load_data():
    lines = []
    with open('{}/driving_log.csv'.format(data_folder)) as csvfile:
        for line in csv.reader(csvfile):
            lines.append(line)
    return process_log_lines(lines)

def process_log_lines(lines):   
    images = []
    measurements = []
    for line in lines:
        skip = False
        row_imgs = []
        for i in range(3): #center, left, right
            path = line[i]
            filename = path.split('/')[-1]
            current_path = '{}/IMG/{}'.format(data_folder, filename)
            if os.path.isfile(current_path):
                image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB) #convert for drive.py
                row_imgs.append(image)
            else:
                skip = True
                i = 2
        if not skip:
            images.extend(row_imgs)
            correction = 0.2
            measurement = float(line[3])
            measurements.append(measurement)
            measurements.append(measurement + correction)
            measurements.append(measurement - correction)
    
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train, y_train = process_log_lines(batch_samples)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_input_paths():
    samples = []
    with open('{}/driving_log.csv'.format(data_folder)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def nvidia_model(input_shape):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))

    model.add(Cropping2D(((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model


def lenet5_model(input_shape):
    model = Sequential()
    model.add(Lambda (lambda x: x/255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5,activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


def main():
    samples = get_input_paths()
    train_samples, validation_samples = train_test_split(samples, test_size=test_size)
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
#     cbs = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
#                                          verbose=1, mode='auto', baseline=None, restore_best_weights=True)]

    ch, row, col = 3, 160, 320 
#     model = lenet5_model((row, col, ch))
    model = nvidia_model((row,col,ch))
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                        validation_data=validation_generator, 
                        nb_val_samples=len(validation_samples), nb_epoch=epochs)
#                        callbacks = cbs)

    model.save('model.h5')

data_folder = './data'
batch_size = 32
test_size = 0.2
# validation_split = 0.2
epochs = 3
main()