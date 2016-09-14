from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import lipnet_input
from confusion_matrix import ConfusionMatrix
from keras.utils import np_utils

"""
problem_name = 'packiging'
nb_epoch = 10

train_set = lipnet_input.get_dataset_images_keras(problem_name,
                                                  'train',
                                                  do_oversampling=False,
                                                  img_size=(28, 28))
train_set.oversample()

validation_set = lipnet_input.get_dataset_images_keras(problem_name,
                                                       'validation',
                                                       do_oversampling=False,
                                                       img_size=(28, 28))
"""


def cnn(train_set, test_set, nb_epoch=10, verbose=True):
    x_train = train_set.x
    x_test = test_set.x

    n_classes = train_set.num_classes

    # train set targets
    y_train = train_set.y
    y_train = np_utils.to_categorical(y_train, n_classes)

    # label smoothing
    eps = 0.1
    y_train = y_train * (1 - eps) + (1 - y_train) * eps / (n_classes - 1.0)

    # validation set targets
    y_test = test_set.y
    y_test = np_utils.to_categorical(y_test, n_classes)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3,
                            border_mode='same',
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # get class weights
    class_weights_balanced = train_set.balanced_class_weights
    class_weights = {}
    for i, weight in enumerate(class_weights_balanced):
        class_weights[i] = weight

    # data augmentation
    datagen = ImageDataGenerator(
        zca_whitening=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )

    datagen.fit(x_train)

    v = 1
    if not verbose:
        v = 0

    model.fit_generator(datagen.flow(x_train, y_train, shuffle=True),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=nb_epoch,
                        #validation_data=(x_test, y_test),
                        verbose=v)

    # y_pred = model.predict_proba(x_train, verbose=0)
    # cf = ConfusionMatrix(y_pred, y_train)
    # print 'Train:'
    # print cf.as_str

    y_pred = model.predict_proba(x_test, verbose=0)
    cf = ConfusionMatrix(y_pred, y_test)

    if verbose:
        print 'Validation:'
        print cf.as_str

    return cf