import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
