from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import TensorBoard
from keras import backend as K
import sys
sys.path.append("game")
import wrapped_flappy_bird as game
import matplotlib.pyplot as plt

img_rows = 80
img_cols = 80
GAMA = 0.9

def build_model():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 8, 8, subsample(4, 4), activation='relu', kernel_initializer = 'he_normal', border_mode='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 4, 4, subsample(2, 2), activation='relu', kernel_initializer = 'he_normal', border_mode='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, subsample(1,1), activation='relu', kernel_initializer = 'he_normal', border_mode='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    f4 = Flatten()(pool3)
    f4 = Dense(256, activation = 'relu')(f4)

    f5 = Dense(2)(f4)

    model = Model(input=inputs, output=f5)
    model.compile(optimizer=Adam(lr = 1.0e-5), loss = mse, metrics=['accuracy'])

    return model

def play():
    model = build_model()
    flappy = game.GameState()

    action = np.array([1])
    observe, reward, terminal = flappy.frame_step(action)
    observe = cv2.cvtColor(cv2.resize(observe, (80, 80)), cv2.COLOR_BGR2GRAY)

    epoch = 0
    while 1==1:
        observe_next, reward_next, terminal_next = flappy.frame_step(action)
        observe = cv2.cvtColor(cv2.resize(observe, (80, 80)), cv2.COLOR_BGR2GRAY)
        Q_s = model.train_on_batch(observe_next)
        if terminal_next:
            target = reward_next
        else:
            target = reward_next + GAMA * np.max(Q_s)
        model.fit(observe, target)
        observe = observe_next
        action = np.argmax(model.predict(observe)[0])
        epoch = epoch+1
        if epoch%100 == 0:
            model.save_weights("./dqn.hdf5")

if __name__ == '__main__':
    play()




