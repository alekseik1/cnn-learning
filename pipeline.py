from keras.datasets import mnist
from preprocessing import ImageScaler, preprocess_categories
from sklearn.pipeline import Pipeline
from model import MainModel

MODEL_NAME = 'model_trained.h5'

if __name__ == '__main__':
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train, y_test = preprocess_categories(y_train), preprocess_categories(y_test)

    pipeline = Pipeline([
       ('scaler', ImageScaler),
       ('regressor', MainModel(save_after_train=True))
    ])
    pipeline.fit(x_train, y_train)
