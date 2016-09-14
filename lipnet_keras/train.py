from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import lipnet_input
from confusion_matrix import ConfusionMatrix

problem_name = 'packiging'

train_set = lipnet_input.get_dataset_vironova_svm(problem_name, 'train', do_oversampling=False)
x = train_set._df[train_set.feature_names].values
y = train_set._df[train_set._class_columns].values

model = Sequential()

model.add(Dense(output_dim=100, input_dim=x.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(output_dim=100, input_dim=100))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(output_dim=y.shape[1]))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

class_weights_balanced = train_set.balanced_class_weights
class_weights = {}
for i, weight in enumerate(class_weights_balanced):
    class_weights[i] = weight

model.fit(x,
          y,
          nb_epoch=500,
          class_weight=class_weights,
          verbose=2)

y_ = model.predict_proba(x, verbose=0)
cf = ConfusionMatrix(y_, y)
print 'Train:'
print cf.as_str

test_set = lipnet_input.get_dataset_vironova_svm(problem_name, 'test', do_oversampling=False)
x = test_set._df[train_set.feature_names].values
y = test_set._df[train_set._class_columns].values

y_ = model.predict_proba(x, verbose=0)
cf = ConfusionMatrix(y_, y)

print 'Test:'
print cf.as_str