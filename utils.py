from keras.models import Input
from keras import Model


def test_layer(layer, data):
    input_layer = Input(shape=data.shape[1:])
    model = Model(inputs=[input_layer], outputs=layer(input_layer))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model.predict(data)
