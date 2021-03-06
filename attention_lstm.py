from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import multiply
"""
copy from: https://github.com/philipperemy/keras-attention-mechanism
"""


# from attention_utils import get_activations, get_data_recurrent

INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
# SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_3d_block(inputs, time_steps, single_attention_vector=False):
    """

    :param inputs:
    :param time_steps:
    :param single_attention_vector:
    :return:
    """
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)                         # (batch_size, input_dim, time_steps)
    a = Reshape((input_dim, time_steps))(a)             # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)      # (batch_size, input_dim, time_steps)  时间序列上全链接层, softmax输出为权重
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)  # (batch_size, time_steps, input_dim)  转回原始结构
    output_attention_mul = multiply([inputs, a_probs])  # (batch_size, time_steps, input_dim)  对应项相乘
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    N = 300000
    # N = 300 -> too few = no training
    # inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    #
    # if APPLY_ATTENTION_BEFORE_LSTM:
    #     m = model_attention_applied_before_lstm()
    # else:
    #     m = model_attention_applied_after_lstm()
    #
    # m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # print(m.summary())
    #
    # m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)
    #
    # attention_vectors = []
    # for i in range(300):
    #     testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    #     attention_vector = np.mean(get_activations(m,
    #                                                testing_inputs_1,
    #                                                print_shape_only=True,
    #                                                layer_name='attention_vec')[0], axis=2).squeeze()
    #     print('attention =', attention_vector)
    #     assert (np.sum(attention_vector) - 1.0) < 1e-5
    #     attention_vectors.append(attention_vector)
    #
    # attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # # plot part.
    # import matplotlib.pyplot as plt
    # import pandas as pd
    #
    # pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
    #                                                                      title='Attention Mechanism as '
    #                                                                            'a function of input'
    #                                                                            ' dimensions.')
    # plt.show()
