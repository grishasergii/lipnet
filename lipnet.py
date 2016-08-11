import sys
import getopt
from lipnet_dataset import DatasetPDAugmented


dataset_name = None
options, remainder = getopt.getopt(sys.argv[1:], '', ['dataset='])
for opt, arg in options:
    if opt in ('--dataset'):
        dataset_name = arg

if dataset_name is None:
    dataset_name = 'train'

problem = 'lamellarity'
dir = '/home/sergii/Documents/microscopic_data/{}/'
path_to_json = dir + '{}_' + dataset_name + '_set.json'
path_to_img = dir + 'images/without_padding/'

def main():
    # create a dataset
    train_set = DatasetPDAugmented(path_to_json.format(problem, problem),
                          path_to_img.format(problem),
                          batch_size=500,
                          num_epochs=1)
    b = train_set.next_batch()
    while b is not None:
        print b.size
        b = train_set.next_batch()
    pass

if __name__ == '__main__':
    main()