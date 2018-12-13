from custom_layers import base_embed_lstm_net, base_embed_cnn_lstm_net, base_multi_channel_net
import keras


def mr_base_model(dataset, vocabulary_size, maxlen):
    # 0.7749  0.7270  0.7720
    # model = base_embed_lstm_net(vocabulary_size)

    # model = base_embed_cnn_lstm_net(vocabulary_size)

    model = base_multi_channel_net(vocabulary_size, time_steps=maxlen)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='auto'
        )
    ]
    history = model.fit(dataset.x_train, dataset.y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=(dataset.x_val, dataset.y_val),
                        callbacks=callbacks,
                        verbose=0)
    return model, history

