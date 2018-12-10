from custom_layers import base_embed_lstm_net


def mr_base_model(dataset, max_word_num):
    model = base_embed_lstm_net(max_word_num)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(dataset.x_train, dataset.y_train,
                        epochs=10,
                        batch_size=128,
                        validation_data=(dataset.x_val, dataset.y_val))
    return model

