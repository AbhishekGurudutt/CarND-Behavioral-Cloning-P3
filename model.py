
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Convolution2D, BatchNormalization
import matplotlib.pyplot as plt
import random


# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


def preprocess(image):
    """
    Function for preprocessing.
    Crops the image
    :param image: Image that has to preprocessed
    :return: preprocessed image
    """
    new_img = image[55:160, :, :]
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img


def hist(input):
    """
    A function to display histogram.
    :return: None
    """
    slots = 50
    samples = len(input) / slots
    hist, slots = np.histogram(input, slots)
    width = 0.7 * (slots[1] - slots[0])
    center = (slots[:-1] + slots[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(input), np.max(input)), (samples, samples), 'k-')
    plt.show()


def nvidia_model(model):
    """
    Nvidia model
    :param model: input is a keras model
    :return: model
    """
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model


def normalize_data(images, angles):

    discard_data = []
    for idx in range(len(angles)):
        if (angles[idx] == 0) or (angles[idx] == 0.25) or (angles[idx] == -0.25):
            discard_data.append(idx)

    length = int(len(discard_data) * 0.6)
    random_discard = random.sample(discard_data, length)
    print("discard data length: ", len(random_discard))
    print("before shape: ", images.shape, angles.shape)
    images = np.delete(images, random_discard, axis=0)
    angles = np.delete(angles, random_discard)

    print("After shape: ", images.shape, angles.shape)
    return images, angles


def read_data():
    """
    Reads images and steering angle required for training
    :return: read images and steering angles after processing
    """
    lines = []
    images = []
    measurements = []

    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        for i in range(3):
            offset = 0.0
            if i == 1:
                offset = 0.25
            elif i == 2:
                offset = -0.25

            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename
            image = cv2.imread(current_path)
            images.append(preprocess(image))
            angle = float(line[3]) + offset
            measurements.append(angle)

            image = cv2.flip(image, 1)
            images.append(preprocess(image))
            measurements.append(angle * -1)

    return np.array(images), np.array(measurements)

x_train, y_train = read_data()
x_train, y_train = normalize_data(x_train, y_train)
hist(y_train)


model = Sequential()

model = nvidia_model(model)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
