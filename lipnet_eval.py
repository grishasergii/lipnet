from lipnet_dataset import DatasetPD, DatasetPDFeatures
from lipnet_tf.model import Model
from lipnet_tf.evaluate import evaluate
from lipnet_tf import FLAGS
import lipnet_architecture as la

def evaluate_real_dataset(problem, dataset):
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_{}_set.json'
    path_to_img = dir + 'images/without_padding/'
    batch_size = 700
    FLAGS.batch_size = batch_size
    epochs = 1
    # create dataset
    dataset = DatasetPD(path_to_json.format(problem, problem, dataset),
                          path_to_img.format(problem),
                          batch_size=batch_size,
                          num_epochs=epochs)

    model = Model(3, la.layer_definitions)
    loss, acc, confusion_matrix = evaluate(dataset, model, do_restore=True)


def evaluate_synthetic_dataset():
    path_to_json = './output/synthetic_examples/synthetic_particles.json'
    path_to_img = './output/synthetic_examples/synthetic_images'
    batch_size = 700
    FLAGS.batch_size = batch_size
    epochs = 1
    # create dataset
    dataset = DatasetPD(path_to_json,
                          path_to_img,
                          batch_size=batch_size,
                          num_epochs=epochs)

    model = Model(3, la.layer_definitions)
    loss, acc, confusion_matrix = evaluate(dataset, model, do_restore=True)


def main(argv=None):
    problem = 'packiging'
    dataset = 'test'
    #print 'Synthetic dataset:'
    #evaluate_synthetic_dataset()
    #print '\nReal dataset:'
    evaluate_real_dataset(problem, dataset)

if __name__ == '__main__':
    main()