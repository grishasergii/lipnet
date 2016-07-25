import tensorflow as tf
import pandas as pd
import numpy as np
import os.path


def read_image_from_disk(input_queue, path):
    """
    Read individual image from disk
    :param input_queue:
    :param path: path to the folder with images
    :return: tensor corresponding to the image, tensor corresponding to labels
    """
    label = input_queue[1]
    file_contents = tf.read_file(path + input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=1)
    # resize image
    w = h = 28
    example = tf.image.resize_images(example, w, h)

    return example, label


def inputs(particles_df, path_to_image, batch_size, shuffle_batch=True):
    """
    Construct inputs for lipnet
    :param particles_df: pandas data frame describing particles
    :param data_dir: Path to the Lipnet images directory
    :param batch_size: Number of images per batch
    :param shuffle_batch: boolean, shuffle examples in batch or not
    :return:
        images: 3D tensor of [batch_size, width, height]. All images have different size
        labels: 1D tensor of [batch_size] with labels
    """

    # https://gist.github.com/eerwitt/518b0c9564e500b4b50f

    # http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

    # Create list of images and corresponding labels
    image_list = particles_df['Image'].values
    for f in image_list:
        if not tf.gfile.Exists(path_to_image + f):
            raise ValueError('Failed to find file: ' + f)

    label_columns = [col for col in list(particles_df) if col.startswith('Label')]
    label_list = particles_df[label_columns].values

    images = tf.convert_to_tensor(image_list,
                                  dtype=tf.string)
    labels = tf.convert_to_tensor(label_list,
                                  dtype=tf.int32)

    # make an input queue that produces the filenames to read and corresponding labels
    input_queue = tf.train.slice_input_producer([images, labels],
                                                #num_epochs=num_epochs,
                                                shuffle=True)

    # read images from disk and corresponding labels in the input queue
    image, label = read_image_from_disk(input_queue, path_to_image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_whitening(image)

    # group examples into batches randomly or not
    if shuffle_batch:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          capacity=2000,
                                                          min_after_dequeue=1000)
    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size
                                                    )

    # display training images in the Tensorboard
    tf.image_summary('images', image_batch, max_images=20)

    return image_batch, label_batch
