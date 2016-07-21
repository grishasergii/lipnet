"""
A module to preprocess images of liposomes
"""
from skimage import io
import pandas as pd
from progressbar import ProgressBar, Percentage, Bar
import shutil
import os



def _remove_padding(path_to_image, output_path, padding):
    """
    Removes padding of a single image and saves output to a new file
    :param path_to_image: full path to an input image
    :param output_path: full path to a file in which output result is saved
    :param padding: integer
    :return: nothing
    """
    if not os.path.isfile(path_to_image):
        print 'Warning: %s not found' % path_to_image
        return
    # read image
    image = io.imread(path_to_image)
    dim = image.shape
    x = dim[0] - padding
    y = dim[1] - padding
    # crop the image
    image_cropped = image[padding:x, padding:y]
    # save cropped image
    io.imsave(output_path, image_cropped)

def remove_padding():
    """
    Remove paddings
    :return: nothing
    """
    problem_folder = 'lamellarity'
    out_dir = '/home/sergii/Documents/microscopic_data/' + problem_folder + '/images/without_padding/'
    in_dir = '/home/sergii/Documents/microscopic_data/' + problem_folder + '/images/particles/'
    path_to_json = '/home/sergii/Documents/microscopic_data/' + problem_folder + '/particles_repaired_2.json'

    print '...Reading dataframe at %s' % path_to_json
    df = pd.read_json(path_to_json)

    print '...Starting processing images'

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    total_count = df.shape[0]
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=total_count).start()
    counter = 0
    for _, row in df.iterrows():
        img_name = row['Image']
        padding = row['Padding']
        _remove_padding(in_dir + img_name,
                        out_dir + img_name,
                        padding)
        counter += 1
        pbar.update(counter)
    pbar.finish()



def main(argv=None):
    remove_padding()


if __name__ == '__main__':
    main()
