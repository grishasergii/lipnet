from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import lipnet_input
from confusion_matrix import ConfusionMatrix
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from model import ModelBasic


class ModelFullyConneted(ModelBasic):

    def build_model(self, input_dim, output_dim):
        self.model.add(Dense(output_dim=100, input_dim=input_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(output_dim=50, input_dim=100))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(output_dim=output_dim))
        self.model.add(Activation('softmax'))

        if self._compile_on_build:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])

    def fit(self, train_set, nb_epoch):
        super(ModelFullyConneted, self).fit(train_set, nb_epoch)
        """
        class_weights_balanced = train_set.balanced_class_weights
        class_weights = {0: 1.2, 1: 1, 2: 1.2}
        for i, weight in enumerate(class_weights_balanced):
           class_weights[i] = weight
        """
        verbose = 0
        if self.verbose:
            verbose = 2
        self.model.fit(self.x_train,
                       self.y_train,
                       nb_epoch=nb_epoch,
                       # class_weight=class_weights,
                       verbose=verbose)

    def preprocess_x(self, x):
        return self._scaler.fit_transform(x)


def main():
    problem_name = 'packiging'

    train_set = lipnet_input.get_dataset_vironova_svm(problem_name, 'train', do_oversampling=False)
    test_set = lipnet_input.get_dataset_vironova_svm(problem_name, 'test', do_oversampling=False)
    #fit(train_set, test_set, nb_epoch=40, verbose=True)

    model = ModelFullyConneted(verbose=True)
    train_set.oversample()
    model.fit(train_set, nb_epoch=40)

    # print confusion matrix of train set
    cf = model.evaluate(train_set)
    print 'Train:'
    print cf.as_str

    # print confusion matrix of test set
    cf = model.evaluate(test_set)
    print 'Test:'
    print cf.as_str

if __name__ == '__main__':
    main()