# order images
# place images of the same class in separate folder

import os
import shutil
import helpers
import pandas as pd



problem = 'packiging'
dir = '/home/sergii/Documents/microscopic_data/{}/'
path_to_json = dir + 'particles_repaired.json'
path_to_img = dir + 'images/without_padding/'

out_dir = dir.format('images_grouped/' + problem + '/') + '{}/'

# create a dataset
df = pd.read_json(path_to_json.format(problem))

# replace class int with names
df['Class'] = df['Class'].replace(to_replace=[3, 4, 5, 7, 8, 10],
                                  value=['Unilamellar', 'Multilamellar', 'Uncertain', 'Empty', 'Full', 'Uncertain'])
labels = df['Class'].unique()

for label in labels:
    helpers.prepare_dir(out_dir.format(label))

total_images = len(df.index)
i = 0
for _, row in df.iterrows():
    i += 1
    print 'Image {} of {}'.format(i, total_images)
    src = os.path.join(path_to_img.format(problem), row['Image'])
    dst = os.path.join(out_dir.format(row['Class']), row['Image'])
    try:
        shutil.copyfile(src, dst)
    except IOError:
        print 'Warning: {} was not copied'.format(row['Image'])


