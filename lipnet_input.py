import pandas as pd
import numpy as np
import json
import os
import re
from scipy import misc
from tf_lipnet import tf_lipnet_train


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
    df = pd.get_dummies(df, prefix='Label', columns=['Class'])
    return df


def main():
    dir = '/home/sergii/Documents/microscopic_data/packiging/'
    path_to_json = dir + 'particles_repaired_2.json'
    path_to_img = dir + 'images/particles/'
    #repair_json(dir + 'particles.json', dir + 'particles_repaired_2.json')


if __name__ == '__main__':
    main()