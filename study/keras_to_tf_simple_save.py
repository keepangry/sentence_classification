#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-2 下午9:51
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : keras_to_tf_simple_save.py
# @Software: PyCharm


import tensorflow as tf
from keras import backend as K
from keras import layers, Input, regularizers
from keras.models import Sequential, Model
from attention_lstm import attention_3d_block
import keras
from datasets import mr_dataset
import os
from global_config import BASE_PATH
from study.keras_model_to_tf_saved_model import build_model


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



if __name__ == "__main__":
    vocabulary_size = 5000
    timesteps = 32
    dataset = mr_dataset(vocabulary_size, timesteps)
    model = model_build_and_fit(dataset, vocabulary_size, timesteps)
    """
    文档：https://tensorflow.google.cn/api_docs/python/tf/saved_model/simple_save
    代码：https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/saved_model/simple_save.py
    simple_save就是封装了默认值的 SavedModelBuilder
    
      signature_def_map = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              signature_def_utils.predict_signature_def(inputs, outputs)
      }
      b = builder.SavedModelBuilder(export_dir)
      b.add_meta_graph_and_variables(
          session,
          tags=[tag_constants.SERVING],
          signature_def_map=signature_def_map,
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
          legacy_init_op=legacy_init_op,
          clear_devices=True)    
    """
    # model.inputs : [<tf.Tensor 'text:0' shape=(?, 32) dtype=int32>]
    tf.saved_model.simple_save(K.get_session(),
                "saved_models/simple_saved_model/",
                inputs={'x': model.inputs[0]},
                outputs={'y': model.outputs[0]})

