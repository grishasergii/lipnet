import pandas as pd
import numpy as np
import json
import os


def repair_json(in_file, out_file):
    '''
    Repairs original JSON files that describe particles. Put comma between JSON objects and enclose the list into square
    brackets
    :param in_file: full path to JSON file to be repaired
    :param out_file: full path file where repaired JSON is dumped
    :return: dumps repaired JSON to a file with specified name
    '''
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
            json_rows.append(''.join(data[start:end+1]))
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


def main():
    path_to_lamellarity = '/home/sergii/Documents/microscopic_data/packiging/particles.json'
    path_to_lamellarity_repaired = '/home/sergii/Documents/microscopic_data/packiging/particles_repaired.json'

    #repair_json(path_to_lamellarity, path_to_lamellarity_repaired)
    with open(path_to_lamellarity_repaired, 'rb') as f:
        p = json.load(f)
    print len(p)
    print p[20001]['Id']



if __name__ == '__main__':
    main()