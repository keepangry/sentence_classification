import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
print(BASE_PATH)


config = tf.ConfigProto()
# config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 占用GPU90%的显存
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)
