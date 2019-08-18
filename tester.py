from keras.datasets import mnist
import keras.backend as K
from keras.layers import Lambda, Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def test_layer(layer, data):
    from keras import Model
    from keras.models import Input
    input_layer = Input(shape=data.shape[1:])
    model = Model(inputs=[input_layer], outputs=layer(input_layer))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data, batch_size=60)


def test_model(data):
    from keras import Model
    from keras.models import Input
    input_layer = Input(shape=data.shape[1:])
    layer = Lambda(lambda x: K.reshape(x, (1, 28*28*60)), output_shape=(1, 28*28*60))(input_layer)
    dense = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=[input_layer], outputs=[dense])
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data, batch_size=60)


def reshape_layer(data):
    tmp = K.reshape(data, (-1, 7, 28, 28))
    return K.reshape(tmp, (-1, 1))


ReshapeLayer = Lambda(lambda data: reshape_layer(data))


if __name__ == '__main__':
    #a = test_layer(ReshapeLayer, x_train)
    #print(a.shape)
    test_model(x_train)
