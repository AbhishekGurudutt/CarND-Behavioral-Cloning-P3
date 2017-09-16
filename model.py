
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Convolution2D, Dropout
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
    new_img = image[55:140, :, :]
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
    model.add(Dropout(0.50))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='tanh'))

    return model


def normalize_data(images, angles):

    discard_data = []
    discard_data_rl = []
    for idx in range(len(angles)):
        if (angles[idx] == 0):
            discard_data.append(idx)
        if (angles[idx] == 0.25) or (angles[idx] == -0.25):
            discard_data_rl.append(idx)

    length = int(len(discard_data) * 0.85)
    random_discard = random.sample(discard_data, length)
    length = int(len(discard_data_rl) * 0.7)
    random_discard_rl = random.sample(discard_data_rl, length)
    print("discard data length straight: ", len(random_discard))
    print("discard data length right left: ", len(random_discard_rl))
    print("before shape: ", images.shape, angles.shape)
    images = np.delete(images, random_discard, axis=0)
    angles = np.delete(angles, random_discard)

    images = np.delete(images, random_discard_rl, axis=0)
    angles = np.delete(angles, random_discard_rl)

    print("After shape: ", images.shape, angles.shape)
    return images, angles

def add_shadow(image):
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    height, width = 66, 200
    x1, y1 = width * np.random.rand(), 0
    x2, y2 = width * np.random.rand(), height
    x, y = np.mgrid[0:height, 0:width]

    msk = np.zeros_like(img[:, :, 1])
    msk[(y - y1) * (x2 - x1) - (y2 - y1) * (x - x1) > 0] = 1

    side = msk == np.random.randint(2)

    shadow = np.random.uniform(low=0.2, high=0.5)

    img[:, :, 1][side] = img[:, :, 1][side] * shadow
    img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def generator(images, angles, batch_size=64):
    batch_images = []
    batch_angles = []
    while True:
        for i in range(batch_size):
            length = len(images)
            idx = np.random.randint(low=0, high=length)
            batch_images.append(add_shadow(images[idx]))
            batch_angles.append(angles[idx])
        yield np.array(batch_images), np.array(batch_angles)
        batch_images = []
        batch_angles = []


def generator_validation(images, angles, batch_size=64):
    batch_images = []
    batch_angles = []
    while True:
        for i in range(batch_size):
            length = len(images)
            idx = np.random.randint(low=0, high=length)
            batch_images.append(images[idx])
            batch_angles.append(angles[idx])
        yield np.array(batch_images), np.array(batch_angles)
        batch_images = []
        batch_angles = []

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

training = generator(x_train, y_train, batch_size=64)
validation = generator_validation(x_train, y_train, batch_size=64)

model.fit_generator(training, samples_per_epoch=x_train.shape[0], verbose=2, nb_epoch=5, show_accuracy=True, validation_data=validation, nb_val_samples=6500)
model.save('model.h5')
