from model import ModelBasic
from keras.layers import Merge, Dense
from keras.utils import np_utils
from confusion_matrix import ConfusionMatrix
import lipnet_input


class ModelCombined(ModelBasic):
    def build_model(self, models, input_dims, output_dim):
        keras_models = [None] * len(models)
        for i, m in enumerate(models):
            m.build_model(input_dims[i], output_dim)
            keras_models[i] = m.model
        merged = Merge(keras_models, mode='concat')
        self.model.add(merged)
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, train_set, nb_epoch):
        raise NotImplemented('Not implemented. Use fit_combined')

    def fit_combined(self, models, train_sets, nb_epoch):
        x_train = [None] * len(train_sets)
        input_dims = [None] * len(train_sets)
        for i, train_set in enumerate(train_sets):
            x_train[i] = train_set.x
            x_train[i] = models[i].preprocess_x(x_train[i])
            if len(x_train[i].shape) == 2:
                input_dims[i] = x_train[i].shape[1]
            else:
                input_dims[i] = x_train[i].shape[1:]

        self.build_model(models, input_dims, train_sets[0].num_classes)
        self.y_train = self.get_y_for_train(train_sets[0])
        self.model.fit(x_train, self.y_train, nb_epoch=nb_epoch)

    def evaluate(self, test_set):
        raise NotImplemented('Not implemented. Use evaluate_combined')

    def evaluate_combined(self, test_sets, models):
        x_test = [None] * len(test_sets)
        for i, test_set in enumerate(test_sets):
            x_test[i] = test_set.x
            x_test[i] = models[i].preprocess_x(x_test[i])

        y = test_sets[0].y
        y = np_utils.to_categorical(y, test_sets[0].num_classes)
        y_pred = self.model.predict_proba(x_test, verbose=0)
        cf = ConfusionMatrix(y_pred, y)
        return cf


def main():
    problem_name = 'packiging'
    train_sets = [None] * 2
    test_sets = [None] * 2
    models = [None] * 2

    # fully connected
    train_sets[0] = lipnet_input.get_dataset_vironova_svm(problem_name, 'train', do_oversampling=False)
    train_sets[0].oversample()
    test_sets[0] = lipnet_input.get_dataset_vironova_svm(problem_name, 'test', do_oversampling=False)
    from model_fully_connected import ModelFullyConneted
    models[0] = ModelFullyConneted(verbose=False, compile_on_build=False)

    # CNN
    train_sets[1] = lipnet_input.get_dataset_images_keras(problem_name, 'train', (28, 28))
    train_sets[1].oversample()
    test_sets[1] = lipnet_input.get_dataset_images_keras(problem_name, 'test', (28, 28))
    from model_cnn import ModelCNN
    models[1] = ModelCNN(verbose=False, compile_on_build=False)

    model_combined = ModelCombined(verbose=True)
    model_combined.fit_combined(models, train_sets, nb_epoch=20)

    # print confusion matrix of train set
    cf = model_combined.evaluate_combined(train_sets, models)
    print 'Train:'
    print cf.as_str

    # print confusion matrix of test set
    cf = model_combined.evaluate_combined(test_sets, models)
    print 'Test:'
    print cf.as_str

if __name__ == '__main__':
    main()
