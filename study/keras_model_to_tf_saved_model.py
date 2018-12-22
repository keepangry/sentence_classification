import tensorflow as tf
from keras import backend as K
from keras import layers, Input, regularizers
from keras.models import Sequential, Model
from attention_lstm import attention_3d_block
import keras
from datasets import mr_dataset
import os
from global_config import BASE_PATH

#
K.set_learning_phase(1)


class SavedModelParams(object):
    def __init__(self):
        self.saved_model_dir = os.path.join(BASE_PATH, 'saved_models/keras_to_tf_saved_model/')
        self.version = '1'
        self.signature_key = 'test_signature'
        self.input_key = 'input_x'
        self.output_key = 'output'
        # self.model_name = 'test_saved_model'
        self.tags = [tf.saved_model.tag_constants.SERVING]


def build_model(vocabulary_size, timesteps):
    """
    可执行的多路一维卷积
    :param vocabulary_size:
    :return:
    """
    text_input = Input(shape=(timesteps, ), dtype='int32', name='text')
    embedded_text = layers.Embedding(vocabulary_size, 64)(text_input)

    # kernel_size = 3
    channel1 = layers.Conv1D(64, 3, padding="same", activation='relu')(embedded_text)
    channel1 = layers.MaxPool1D(4)(channel1)

    channel2 = layers.Conv1D(64, 4, padding="same", activation='relu')(embedded_text)
    channel2 = layers.MaxPool1D(4)(channel2)

    channel3 = layers.Conv1D(64, 5, padding="same", activation='relu')(embedded_text)
    channel3 = layers.MaxPool1D(4)(channel3)

    concatenated = layers.concatenate([channel1, channel2, channel3], axis=-1)
    concatenated = layers.LSTM(64)(concatenated)

    output = layers.Dense(64, activation='relu')(concatenated)
    output = layers.Dense(1, activation='sigmoid')(output)
    model = Model(text_input, output)
    return model


def save_model(model, saved_model_params):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={saved_model_params.input_key: model.input},
        outputs={saved_model_params.output_key: model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(saved_model_params.saved_model_dir),
        tf.compat.as_bytes(saved_model_params.version))

    if tf.gfile.Exists(export_path):
        tf.gfile.DeleteRecursively(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=saved_model_params.tags,
        signature_def_map={
            saved_model_params.signature_key: signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()


def model_build_and_fit(dataset, vocabulary_size, timesteps):
    model = build_model(vocabulary_size, timesteps)
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
                        verbose=2)
    return model


def tf_load_model(sess, params):
    meta_graph_def = tf.saved_model.loader.load(sess, saved_model_params.tags,
                                                os.path.join(params.saved_model_dir, params.version))
    # 从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name
    x_tensor_name = signature[params.signature_key].inputs[params.input_key].name
    y_tensor_name = signature[params.signature_key].outputs[params.output_key].name

    # 获取tensor 并inference
    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    return sess, x, y


if __name__ == "__main__":
    vocabulary_size = 5000
    timesteps = 32
    dataset = mr_dataset(vocabulary_size, timesteps)
    model = model_build_and_fit(dataset, vocabulary_size, timesteps)

    saved_model_params = SavedModelParams()
    # save keras model
    save_model(model, saved_model_params)

    # sess
    sess = tf.Session(graph=tf.Graph())

    # load model
    sess, x, y = tf_load_model(sess, saved_model_params)

    # predict
    # _x 实际输入待inference的data
    result = sess.run(y, feed_dict={x: [dataset.x_val[0]]})

    # 第一个0代表批量预测的第0条
    # 预测的y是个向量，此时为1维向量
    print(result[0])

