import pickle as cp
import numpy as np
from sliding_window import sliding_window
import keras

def dataloader(DATASET):
    if DATASET == 'opportunity':
        filename = "./Opp113.p"
        SLIDING_WINDOW_LENGTH = 1.
        SLIDING_WINDOW_STEP = 0.5
        NB_SENSOR_CHANNELS = 113
        freq = 24.
        NUM_CLASSES = 18
        normal_loc = 0.5

    elif DATASET == 'dg':
        filename = "./dg.p"
        SLIDING_WINDOW_LENGTH = 1.0
        SLIDING_WINDOW_STEP = 0.5
        NB_SENSOR_CHANNELS = 9
        freq = 64.
        NUM_CLASSES = 2
        normal_loc = 0.

    elif DATASET == 'wetlab':
        filename = './wetlab.p'
        SLIDING_WINDOW_LENGTH = 5.0
        SLIDING_WINDOW_STEP = 0.5
        NB_SENSOR_CHANNELS = 3
        freq = 50.
        NUM_CLASSES = 9
        normal_loc = 0.

    elif DATASET == '50salads':
        filename = './50salads.p'
        SLIDING_WINDOW_LENGTH = 5.12
        NB_SENSOR_CHANNELS = 30
        SLIDING_WINDOW_STEP = 0.5
        freq = 50.
        NUM_CLASSES = 10
        normal_loc = 0.

    data = cp.load(open(filename, 'rb'))
    if len(data) < 3:
        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[1]
    else:
        assert len(data) == 3
        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    print("... reading instances: train {0} {1}, val {2} {3}, test {4} {5}".format(
        X_train.shape, y_train.shape,
        X_val.shape, y_val.shape,
        X_test.shape, y_test.shape))

    assert NB_SENSOR_CHANNELS == X_train.shape[1]
    assert NUM_CLASSES == len(np.unique(y_train)), '{} != {}'.format(NUM_CLASSES, len(np.unique(y_train)))

    SLIDING_WINDOW_LENGTH = int(SLIDING_WINDOW_LENGTH*freq)
    SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_STEP*freq)

    return X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, normal_loc, NB_SENSOR_CHANNELS

