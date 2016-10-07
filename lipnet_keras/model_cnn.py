from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import lipnet_input
from model import ModelBasic
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import helpers
from dataset.dataset_images import DatasetImages
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.misc import imsave


class VisualizerBasic:
    def __init__(self):
        self.layer_dict = {}

    @staticmethod
    def deprocess_image(x):
        """
        util function to convert a tensor into a valid image
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        :param x: numpy 2d array
        :return: numpy 2d array
        """
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')

        return x


class VisualizerFeatures(VisualizerBasic):

    def visualize_model(self, model, out_path, vis_size):
        helpers.prepare_dir(out_path, empty=True)
        # remove all dropout layers
        for layer in model.layers:
            if type(layer) is Dropout:
                model.layers.remove(layer)

        for layer in model.layers:
            if type(layer) is Convolution2D:
                self.visualize_layer(layer, model, 32, out_path, vis_size)

    def visualize_layer(self, layer, model, keep_filters, out_path, vis_size):
        layer_output = layer.output
        num_filters = layer.nb_filter
        filters = []
        img_width = vis_size[0]
        img_height = vis_size[1]

        for filter_index in xrange(num_filters):
            loss = K.mean(K.mean(layer_output[:, filter_index, :, :]))

            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, model.layers[0].input)[0]

            # normalization trick: we normalize the gradient
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

            # this function returns the loss and grads given the input picture
            iterate = K.function([model.layers[0].input], [loss, grads])

            # step size for gradient ascent
            step = 1.

            input_img_data = np.random.random((1, 1, img_width, img_height)) * 20 + 128.

            for i in xrange(50):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                if loss_value <= 0:
                    break

            img = self.deprocess_image(input_img_data[0])
            filters.append((img, loss_value))

        filters.sort(key=lambda x: x[1], reverse=True)
        filters = filters[:keep_filters]

        # get number of grid rows and columns
        grid_r, grid_c = helpers.get_grid_dim(keep_filters)

        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))


        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = filters[l][0][:, :, 0]
            # put it on the grid
            ax.imshow(img, interpolation='bicubic', cmap='Greys')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        out_path = os.path.join(out_path, layer.name) + '.png'
        plt.savefig(out_path, bbox_inches='tight')

        """
        margin = 5
        width = grid_c * img_width + (grid_c - 1) * margin
        height = grid_r * img_height + (grid_r - 1) * margin
        stitched_filters = np.zeros((width, height, 1))
        ix = 0
        for i in xrange(grid_r):
            for j in xrange(grid_c):
                print ix
                img, _ = filters[ix]
                stitched_filters
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
                ix += 1

        helpers.prepare_dir(out_path, empty=True)
        out_path = os.path.join(out_path, layer.name) + '.png'
        imsave(out_path, stitched_filters)
        """


class ModelCNNBasic(ModelBasic):
    def fit(self, train_set, test_set, nb_epoch):
        super(ModelCNNBasic, self).fit(train_set, test_set, nb_epoch)
        # data augmentation
        datagen = ImageDataGenerator(
            zca_whitening=False,
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.2,
            zoom_range=0.2
        )

        datagen.fit(self.x_train)

        verbose = 1
        if not self.verbose:
            verbose = 0

        if test_set is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            checkpoint_path = 'output/checkpoint/{}'.format(self.name)
            helpers.prepare_dir(checkpoint_path, empty=True)
            checkpoint_path = os.path.join(checkpoint_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
            checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=False)
            callbacks = [early_stopping, checkpoint]
            self.model.fit_generator(datagen.flow(self.x_train,
                                                  self.y_train,
                                                  shuffle=True),
                                     samples_per_epoch=self.x_train.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(self.x_test, self.y_test),
                                     callbacks=callbacks,
                                     verbose=verbose,
                                     )
        else:
            self.model.fit_generator(datagen.flow(self.x_train,
                                                  self.y_train,
                                                  shuffle=True),
                                     samples_per_epoch=self.x_train.shape[0],
                                     nb_epoch=nb_epoch,
                                     verbose=verbose
                                     )

    def visualize_weights(self, plot_dir):
        for layer in self.model.layers:
            if isinstance(layer, Convolution2D):
                weights = layer.get_weights()[0]
                self.plot_conv_weights(weights, layer.name, plot_dir)

    def visualize_conv_image(self, image, plot_dir):
        x = image.reshape((1,) + image.shape)
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, Convolution2D):
                get_layer_output = K.function([self.model.layers[0].input, K.learning_phase()],
                                              [layer.output])
                layer_output = get_layer_output([x, 0])[0]
                self.plot_conv_output(layer_output, layer.name, plot_dir)

    @staticmethod
    def plot_conv_weights(weights, name, plot_dir, channels_all=True):
        """
        Plots convolutional filters
        :param weights: numpy array of rank 4
        :param name: string, name of convolutional layer
        :param channels_all: boolean, optionalr
        :return: nothing, plots are saved on the disk
        """
        # make path to output folder
        plot_dir = os.path.join(plot_dir, name)

        # create directory if does not exist, otherwise empty it
        helpers.prepare_dir(plot_dir, empty=True)

        w_min = np.min(weights.flatten())
        w_max = np.max(weights.flatten())

        channels = [1]
        # make a list of channels if all are plotted
        if channels_all:
            channels = range(weights.shape[1])

        # get number of convolutional filters
        num_filters = weights.shape[0]

        # get number of grid rows and columns
        grid_r, grid_c = helpers.get_grid_dim(num_filters)

        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

        # iterate channels
        for channel in channels:
            # iterate filters inside every channel
            for l, ax in enumerate(axes.flat):
                # get a single filter
                img = weights[l, channel, :, :]
                # put it on the grid
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
            # save figure
            plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

    @staticmethod
    def plot_conv_output(conv_img, name, plot_dir):
        """
        Makes plots of results of performing convolution
        :param conv_img: numpy array of rank 4
        :param name: string, name of convolutional layer
        :return: nothing, plots are saved on the disk
        """
        # make path to output folder
        plot_dir = os.path.join(plot_dir, name)

        # create directory if does not exist, otherwise empty it
        helpers.prepare_dir(plot_dir, empty=True)

        w_min = np.min(conv_img)
        w_max = np.max(conv_img)

        # get number of convolutional filters
        num_filters = conv_img.shape[1]

        # get number of grid rows and columns
        grid_r, grid_c = helpers.get_grid_dim(num_filters)

        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

        # iterate filters
        for l, ax in enumerate(axes.flat):
            # get a single image
            img = conv_img[0, l, :, :]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


class ModelLipnet4(ModelCNNBasic):
    def build_model(self, input_dim, output_dim):
        trainable = True
        self.model.add(Convolution2D(32, 3, 3,
                                     border_mode='same',
                                     input_shape=input_dim,
                                     trainable=trainable))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu', trainable=trainable))

        self.model.add(Convolution2D(32, 3, 3, trainable=trainable))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu', trainable=trainable))
        self.model.add(MaxPooling2D(pool_size=(2, 2), trainable=trainable))
        self.model.add(Dropout(0.25, trainable=trainable))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same', trainable=trainable))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu', trainable=trainable))
        self.model.add(Convolution2D(64, 3, 3, trainable=True))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu', trainable=True))
        self.model.add(MaxPooling2D(pool_size=(2, 2), trainable=True))
        self.model.add(Dropout(0.25, trainable=True))

        if self._include_top:
            self.model.add(Flatten())
            self.model.add(Dense(512))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))

            self.model.add(Dense(output_dim))
            self.model.add(BatchNormalization())
            self.model.add(Activation('softmax'))

            if self._compile_on_build:
                optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                self.model.compile(loss='categorical_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])


class ModelLipnet6(ModelCNNBasic):
    def build_model(self, input_dim, output_dim):
        h = 5
        trainable = False
        # 1st conv block
        self.model.add(Convolution2D(32, 7, 7, input_shape=input_dim, border_mode='same', activation='relu', trainable=trainable))
        self.model.add(Convolution2D(32, 7, 7, activation='relu', trainable=trainable))
        self.model.add(MaxPooling2D(pool_size=(2, 2), trainable=trainable))
        self.model.add(Dropout(0.25, trainable=trainable))

        # 2nd conv block
        self.model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same', trainable=trainable))
        self.model.add(Convolution2D(64, 5, 5, activation='relu', trainable=trainable))
        self.model.add(MaxPooling2D(pool_size=(2, 2), trainable=trainable))
        self.model.add(Dropout(0.25, trainable=trainable))

        # 3rd conv block
        self.model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=trainable))
        self.model.add(Convolution2D(128, 3, 3, activation='relu', trainable=trainable))
        self.model.add(MaxPooling2D(pool_size=(2, 2), trainable=trainable))
        self.model.add(Dropout(0.25, trainable=trainable))

        if self._include_top:
            self.model.add(Flatten())

            self.model.add(Dense(512))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))

            self.model.add(Dense(128))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))

            self.model.add(Dense(output_dim))
            self.model.add(Activation('softmax'))

            if self._compile_on_build:
                optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                self.model.compile(loss='categorical_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])


def train(model, img_size, with_padding, nb_epoch):
    problem_name = 'packiging'

    train_set = lipnet_input.get_dataset_images_keras(problem_name, 'train', img_size, with_padding=with_padding)
    test_set = lipnet_input.get_dataset_images_keras(problem_name, 'test', img_size, with_padding=with_padding)

    # model = ModelLipnet4(verbose=True, name='lipnet6_{}'.format(problem_name))
    train_set.oversample()
    model.fit(train_set, test_set, nb_epoch=nb_epoch)

    # print confusion matrix of train set
    cf = model.evaluate(train_set)
    print 'Train:'
    print cf.as_str

    # print confusion matrix of test set
    cf = model.evaluate(test_set)
    print 'Test:'
    print cf.as_str

    model_name = 'lipnet6'
    model.save('output/models/{}_model_{}.h5'.format(problem_name, model_name))


def visualize(model):
    problem_name = 'lamellarity'

    #model = ModelLipnet4(verbose=True)
    #model.restore('output/models/model_cnn_deep.h5')

    model.visualize_weights('output/figures/{}/cnn_deep/conv_weights/'.format(problem_name))
    image_name = '539193.jpg'
    image_path = os.path.join(lipnet_input.path_to_img.format(problem_name), image_name)
    image = DatasetImages.read_image(image_path, (28, 28))
    model.visualize_conv_image(image, 'output/figures/{}/cnn_deep/conv_output/'.format(problem_name))


def visualize_features(model, problem_name):
    out_path = 'output/figures/{}/features/'
    im_size = (52, 52)
    K.set_learning_phase(0)

    if model is None:
        model = ModelLipnet4(verbose=True, compile_on_build=False, include_top=True)
        model.build_model((1, im_size[0], im_size[1]), 3)
        model.restore('/home/sergii/Documents/Thesis/lipnet/output/models/{}_model_lipnet6.h5'.format(problem_name))

    vis = VisualizerFeatures()
    vis.visualize_model(model.model, out_path.format(problem_name), im_size)


if __name__ == '__main__':
    problem_name = 'packiging'

    model = ModelLipnet4(verbose=True, name='lipnet6_{}'.format(problem_name))
    img_size = (28, 28)
    train(model, img_size=img_size, with_padding=True, nb_epoch=100)
    #visualize()
    #visualize_features(None, 'packiging')