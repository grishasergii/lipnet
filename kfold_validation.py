from kfold import KFold
from lipnet_dataset import DatasetPD, DatasetPDAugmented
from lipnet_tf import train
import lipnet_architecture as la
from lipnet_tf.model import Model
from datetime import datetime


def do_validation(k, path_to_json, path_to_img, do_augmentation=False, smote_rates=None, verbose=True):
    batch_size = 300
    kfold = KFold(k, path_to_json, path_to_img)
    for i in xrange(k):
        if verbose:
            print '{}: Fold {} of {}'.format(datetime.now(), i + 1, k)
        train_df, test_df = kfold.get_datasets(i)
        if do_augmentation and smote_rates is not None:
            train_set = DatasetPDAugmented(train_df, batch_size=batch_size, num_epochs=30,
                                           smote_rates=smote_rates, verbose=verbose)
        else:
            train_set = DatasetPD(train_df, batch_size=batch_size, num_epochs=30, verbose=verbose)
        test_set = DatasetPD(test_df, batch_size=batch_size, num_epochs=1, verbose=verbose)
        model = Model(3, la.layer_definitions)
        stats = train.train(train_set, model, test_set, verbose=True)
        print stats


if __name__ == '__main__':
    problem = 'packiging'
    path_to_json = '/home/sergii/Documents/microscopic_data/{}/particles_repaired.json'.format(problem)
    path_to_img = '/home/sergii/Documents/microscopic_data/{}/images/without_padding/'.format(problem)
    do_validation(5, path_to_json, path_to_img, do_augmentation=False, smote_rates=[200, 200])