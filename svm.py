import lipnet_input
from sklearn.svm import SVC
import numpy as np
from confusion_matrix import ConfusionMatrix


def svm(train_set, test_set):
    clf = SVC(class_weight='balanced')

    x = train_set._df[train_set.feature_names].values
    y = np.argmax(train_set._df[train_set._class_columns].values, axis=1)

    clf.fit(x, y)

    x = test_set._df[test_set.feature_names].values
    y = test_set._df[test_set._class_columns].values

    y_ = clf.predict(x)
    pred = np.zeros([len(y_), len(test_set._class_columns)])
    for i, j in enumerate(y_):
        pred[i, j] = 1

    cf = ConfusionMatrix(pred, y, class_names=test_set.class_names)
    return cf


def main():
    problem_name = 'packiging'

    train_set = lipnet_input.get_dataset_vironova_svm(problem_name=problem_name,
                                                      set_name='train',
                                                      do_oversampling=False,
                                                      batch_size=None)

    test_set = lipnet_input.get_dataset_vironova_svm(problem_name=problem_name,
                                                     set_name='test',
                                                     do_oversampling=False,
                                                     batch_size=None)

    cf = svm(train_set, test_set)
    print cf.as_str


if __name__ == '__main__':
    main()