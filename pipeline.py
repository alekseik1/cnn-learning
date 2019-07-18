from keras.datasets import mnist
from preprocessing import preprocess_data
from model import MainModel

MODEL_NAME = 'model_trained.h5'

if __name__ == '__main__':
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)


    



    model = MainModel()
    model.fit(x_train, y_train)
    model.save(MODEL_NAME)
