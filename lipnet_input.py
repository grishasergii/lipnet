from __future__ import division
import pandas as pd
import json
import os
import os.path
import re
import math


def repair_json(in_file, out_file):
    """
    Repairs original JSON files that describe particles. Put comma between JSON objects and enclose the list into square
    brackets
    :param in_file: full path to JSON file to be repaired
    :param out_file: full path file where repaired JSON is dumped
    :return: dumps repaired JSON to a file with specified name
    """
    with open(in_file, 'rb') as f:
        data = f.readlines()

    # delete all \n characters
    data = map(lambda x: x.rstrip(), data)

    # make one JSON object per row
    json_rows = []
    start = -1
    while True:
        try:
            start = data.index('{', start + 1)
            end = data.index('}', start)
            row = ''.join(data[start:end+1])
            row = re.sub("\"Bmp\"", "\"Image\"", row)
            row = re.sub(".bmp", ".jpg", row)
            json_rows.append(row)
            start = end
        except ValueError:
            break

    # join all JSON objects into one comme delimited string enclosed in square brackets
    data_join_str = "[" + ','.join(json_rows) + "]"

    # create JSON object
    repaired_json = json.loads(data_join_str)

    # remove out_file if it exists
    try:
        os.remove(out_file)
    except OSError:
        pass
    except IOError:
        pass

    with open(out_file, 'w+') as f:
        json.dump(repaired_json, f)


def get_particles_df(path):
    """
    Read JSON that describes particles into a pandas data frame
    :param path: full path to JSON file
    :return: data frame
    """
    df = pd.read_json(path)
    # replace integers with class names for better readability
    df['Class'] = df['Class'].replace(to_replace=[3, 4, 5, 7, 8, 10],
                                      value=['Unilamellar', 'Multilamellar', 'Uncertain', 'Empty', 'Full', 'Uncertain'])
    # present class captions as one hot encoding
    df = pd.concat([df, pd.get_dummies(df['Class'], prefix='Label')], axis=1)
    return df


def make_train_validation_test_sets(path_to_json, out_dir, path_to_images,
                                    train_fraction=0.6,
                                    validation_fraction=0.2,
                                    test_fraction=0.2,
                                    do_print=False):
    """
    Splits entire dataset into train, test and validation sets
    :param path_to_json: string, full path to initial dataset
    :param out_dir: string, path to the output durectory where sets are saved
    :param path_to_images: string,, path to images
    :param train_fraction:
    :param validation_fraction:
    :param test_fraction:
    :param do_print: boolean, print messages to console or not
    :return: nothing
    """
    assert train_fraction + validation_fraction + test_fraction == 1, 'Sum of subsets fractions must be 1'
    df = pd.read_json(path_to_json)
    # one-hot encode labels
    df['Class'] = df['Class'].replace(to_replace=[3, 4, 5, 7, 8, 10],
                                      value=['Unilamellar', 'Multilamellar', 'Uncertain', 'Empty', 'Full', 'Uncertain'])


    # present class captions as one hot encoding
    df = pd.concat([df, pd.get_dummies(df['Class'], prefix='Label')], axis=1)

    # Check that all images in dataframe have corresponding file on the disk
    for index, row in df.iterrows():
        if not os.path.isfile(path_to_images + row['Image']):
            print '{} image was not found. This example will be deleted'.format(row['Image'])
            df.drop(index, inplace=True)

    # prepare new dataframes
    df_train = pd.DataFrame()
    df_validation = pd.DataFrame()
    df_test = pd.DataFrame()

    if do_print:
        print '----------\nEntire set:\n', df['Class'].value_counts()

    class_counts = df['Class'].value_counts().to_dict()
    for label, count in class_counts.iteritems():
        df_test = pd.concat([df_test, df[df['Class'] == label].sample(frac=test_fraction)])
        df = df[~df.index.isin(df_test.index)]

        validation_fraction_adjusted = validation_fraction / (1 - test_fraction)
        df_validation = pd.concat([df_validation, df[df['Class'] == label].sample(frac=validation_fraction_adjusted)])
        df = df[~df.index.isin(df_validation.index)]

        df_train = pd.concat([df_train, df[df['Class'] == label]])
        df = df[~df.index.isin(df_train.index)]

    if do_print:
        print '----------\nTrain set:\n', df_train['Class'].value_counts()
        print '----------\nValidation set:\n', df_validation['Class'].value_counts()
        print '----------\nTest set:\n', df_test['Class'].value_counts()

    # remove out_file if it exists
    filenames = ['train_set.json', 'test_set.json', 'validation_set']
    for f in filenames:
        try:
            os.remove(out_dir + f)
        except OSError:
            pass
        except IOError:
            pass

    df_train.to_json(out_dir + 'train_set.json')
    df_validation.to_json(out_dir + 'validation_set.json')
    df_test.to_json(out_dir + 'test_set.json')


def main():
    problems = ['packiging', 'lamellarity']
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + 'particles_repaired.json'
    path_to_images = dir + 'images/particles/'
    out_path = dir + '{}_'

    for p in problems:
        make_train_validation_test_sets(path_to_json.format(p),
                                        out_path.format(p, p),
                                        path_to_images.format(p),
                                        do_print=True)

    #repair_json(dir + 'particles.json', dir + 'particles_repaired_2.json')


if __name__ == '__main__':
    main()