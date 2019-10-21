from keras.layers import *
from keras.models import *


def attention_3d_block(inputs):
    time_steps = K.int_shape(inputs)[1]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def stacked_lstm(inputs, units, layers):
    lstm = BatchNormalization()(inputs)
    for i in range(layers - 1):
        lstm = LSTM(units, return_sequences=True)(lstm)
    lstm = LSTM(units)(lstm)
    lstm_norm = BatchNormalization()(lstm)
    return lstm_norm


def stacked_dense(inputs, units, layers):
    dense = BatchNormalization()(inputs)
    for i in range(layers):
        dense = Dense(units, activation='tanh')(dense)
    dense_norm = BatchNormalization()(dense)
    return dense_norm
