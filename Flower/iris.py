import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from sklearn.datasets import load_iris


def load_data():
    iris = load_iris()
    data, target = iris.data, iris.target

    X_train = data.astype('float32')
    X_test = data.astype('float32')

    y_train = np_utils.to_categorical(target, 3)
    y_test = np_utils.to_categorical(target, 3)

    X_train = np.reshape(X_train, (150, 4))
    X_test = np.reshape(X_test, (150, 4))

    return [X_train, X_test, y_train, y_test]

def init_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


def run_network(epochs, batch):
    X_train, X_test, y_train, y_test = load_data()
    model = init_model()
    print('Training model...')

    model.fit(X_train, y_train,nb_epoch=epochs, batch_size=batch)
    score = model.evaluate(X_test, y_test, batch_size=15)
    print("Network's test score [loss, accuracy]: {0}".format(score))
    return model


iris = load_iris()
print(iris['target_names'])
run_network(650, 10)