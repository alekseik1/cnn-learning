class Config:
    images_directory = 'debug_imgs'

    bag_size = 32


    classifier_activation = 'sigmoid'
    classifier_loss = 'binary_crossentropy'
    classifier_metrics = 'accuracy'

    decoder_loss = 'binary_crossentropy'

    loss_weights = {'classifier_output': 1.0, 'decoded_output': 1.0}

    optimizer = 'adadelta'
