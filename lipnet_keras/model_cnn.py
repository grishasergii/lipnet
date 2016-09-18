from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
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
from keras.callbacks import EarlyStopping


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
        )

        datagen.fit(self.x_train)

        verbose = 1
        if not self.verbose:
            verbose = 0

        callbacks = []
        validation_data = ()
        if test_set is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            validation_data = (self.x_test,
                               self.y_test)
            callbacks = [early_stopping]

        self.model.fit_generator(datagen.flow(self.x_train,
                                              self.y_train,
                                              shuffle=True),
                                 samples_per_epoch=self.x_train.shape[0],
                                 nb_epoch=nb_epoch,
                                 validation_data=validation_data,
                                 callbacks=callbacks,
                                 verbose=verbose)

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


class ModelCNN_Deep(ModelCNNBasic):
    def build_model(self, input_dim, output_dim):
        self.model.add(Convolution2D(32, 3, 3,
                                     border_mode='same',
                                     input_shape=input_dim))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(output_dim))
        self.model.add(Activation('softmax'))

        if self._compile_on_build:
            optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])


def train():
    problem_name = 'lamellarity'

    train_set = lipnet_input.get_dataset_images_keras(problem_name, 'train', (28, 28))
    test_set = lipnet_input.get_dataset_images_keras(problem_name, 'test', (28, 28))

    model = ModelCNN_Deep(verbose=True)
    train_set.oversample()
    model.fit(train_set, test_set, nb_epoch=100)

    # print confusion matrix of train set
    cf = model.evaluate(train_set)
    print 'Train:'
    print cf.as_str

    # print confusion matrix of test set
    cf = model.evaluate(test_set)
    print 'Test:'
    print cf.as_str

    model.save('output/models/model_cnn_deep.h5')


def visualize():
    problem_name = 'lamellarity'

    model = ModelCNN_Deep(verbose=True)
    model.restore('output/models/model_cnn_deep.h5')

    model.visualize_weights('output/figures/{}/cnn_deep/conv_weights/'.format(problem_name))
    image_name = '539193.jpg'
    image_path = os.path.join(lipnet_input.path_to_img.format(problem_name), image_name)
    image = DatasetImages.read_image(image_path, (28, 28))
    model.visualize_conv_image(image, 'output/figures/{}/cnn_deep/conv_output/'.format(problem_name))


if __name__ == '__main__':
    #train()
    visualize()