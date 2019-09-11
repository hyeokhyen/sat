from keras.models import Sequential
from keras.layers import Conv2D, Reshape, Dropout, LSTM, Dense, Lambda, Activation
from keras import backend as K

'''
CNN + LSTM model
'''

def create_model(NUM_FILTERS=64, FILTER_SIZE=5,
    NUM_UNITS_DENSE=128,
    NUM_CLASSES=18, 
    ACTIVATION='relu',
    DROPOUT_RATE=0.5,
    KERNEL_INIT='glorot_uniform', BIAS_INIT='zeros',
    INPUT_SHAPE=(24, 113, 1)):

    # define the network
    # no pooling layer
    model = Sequential()
    # layer 2
    model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, 1),
        activation=ACTIVATION,
        kernel_initializer=KERNEL_INIT,
        bias_initializer=BIAS_INIT,
        input_shape=INPUT_SHAPE))
    # layer 3
    model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, 1),
        kernel_initializer=KERNEL_INIT,
        bias_initializer=BIAS_INIT,
        activation=ACTIVATION))
    # layer 4
    model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, 1),
        kernel_initializer=KERNEL_INIT,
        bias_initializer=BIAS_INIT,
        activation=ACTIVATION))
    # layer 5
    model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, 1),
        kernel_initializer=KERNEL_INIT,
        bias_initializer=BIAS_INIT,
        activation=ACTIVATION))
    # reshape for lstm input
    model.add(Reshape((8, -1)))
    # layer 6
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(NUM_UNITS_DENSE, 
        return_sequences=True))
    # layer 7
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(NUM_UNITS_DENSE, 
        return_sequences=False))
    # layer 8
    model.add(Dense(NUM_CLASSES, 
        kernel_initializer=KERNEL_INIT,
        bias_initializer=BIAS_INIT,
        activation='linear'))
    # softmax
    model.add(Activation('softmax'))

    return model

