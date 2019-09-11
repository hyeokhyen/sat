# Copyright (c) 2019, Hyeokhyen Kwon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
#   Kwon, H., Abowd, G. D., & Plötz, T. (2019, September). 
#   Handling Annotation Uncertainty in Human Activity Recognition
#   In Proceedings of the 2019 ACM International Symposium on Wearable Computers
#   (pp. x-y). ACM.
#
#   Hyeok Kwon '19
#   hyeokhyen@gatech.edu
#

#----------------------------------------------------------
isTrain = True
DATASET = 'wetlab' # opportunity, dg, wetlab, 50salads
#----------------------------------------------------------

print ('=== ConvLSTM-SAT for {} ==='.format(DATASET))

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataloader import dataloader
from cnnlstm import create_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf_config=tf.ConfigProto()
# dynamically grow the memory used on the GPU
tf_config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
# nothing gets printed in Jupyter, only if you run it standalone
tf_config.log_device_placement = False
sess = tf.Session(config=tf_config)
set_session(sess)
import keras

X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, normal_loc, NB_SENSOR_CHANNELS = dataloader(DATASET)

if isTrain:
    model = create_model(
        NUM_CLASSES=NUM_CLASSES,
        INPUT_SHAPE=(
            SLIDING_WINDOW_LENGTH,
            NB_SENSOR_CHANNELS,
            1))
    print (model.summary())

    loss = keras.losses.binary_crossentropy
    optimizer = keras.optimizers.RMSprop(lr=1e-3)

    model.compile(loss=loss, optimizer=optimizer)

    # training batch generation
    from batch_generator import batch_generator
    training_generator = batch_generator(X_train, y_train,
        NUM_CLASSES=NUM_CLASSES,
        NB_SENSOR_CHANNELS=NB_SENSOR_CHANNELS,
        SLIDING_WINDOW_LENGTH=SLIDING_WINDOW_LENGTH,
        SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
    print ('... generated training generator')

    validation_generator = batch_generator(X_val, y_val,
        NUM_CLASSES=NUM_CLASSES,
        NB_SENSOR_CHANNELS=NB_SENSOR_CHANNELS,
        SLIDING_WINDOW_LENGTH=SLIDING_WINDOW_LENGTH,
        SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP,     
        is_validate=True)
    print ('... generated validation generator')

    # start training
    model.fit_generator(generator=training_generator,
        validation_data=validation_generator,
        epochs=2,
        use_multiprocessing=False,
        workers=4,
        verbose=2)

# evaluate
import numpy as np
from sliding_window import sliding_window

X_train = sliding_window(X_train, 
    (SLIDING_WINDOW_LENGTH, X_train.shape[1]),
    (SLIDING_WINDOW_STEP, 1))
X_train = X_train.astype(np.float32)
X_train = X_train.reshape(-1, 
    SLIDING_WINDOW_LENGTH, 
    NB_SENSOR_CHANNELS, 
    1)

X_val = sliding_window(X_val, 
    (SLIDING_WINDOW_LENGTH, X_val.shape[1]),
    (SLIDING_WINDOW_STEP, 1))
X_val = X_val.astype(np.float32)
X_val = X_val.reshape(-1, 
    SLIDING_WINDOW_LENGTH, 
    NB_SENSOR_CHANNELS, 
    1)

X_test = sliding_window(X_test, 
    (SLIDING_WINDOW_LENGTH, X_test.shape[1]),
    (SLIDING_WINDOW_STEP, 1))
X_test = X_test.astype(np.float32)
X_test = X_test.reshape(-1, 
    SLIDING_WINDOW_LENGTH, 
    NB_SENSOR_CHANNELS, 
    1)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_train = np.asarray([[i[-1]] 
    for i in sliding_window(y_train, 
    (SLIDING_WINDOW_LENGTH, y_train.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_train.shape[1]))
y_train = np.argmax(y_train, axis=1)

y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_val = np.asarray([[i[-1]] 
    for i in sliding_window(y_val, 
    (SLIDING_WINDOW_LENGTH, y_val.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_val.shape[1]))
y_val = np.argmax(y_val, axis=1)

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_test = np.asarray([[i[-1]] 
    for i in sliding_window(y_test, 
    (SLIDING_WINDOW_LENGTH, y_test.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_test.shape[1]))
y_test = np.argmax(y_test, axis=1)

print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
print(" ..after sliding window (val): inputs {0}, targets {1}".format(X_val.shape, y_val.shape))
print(" ..after sliding window (test): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

# load model
if not isTrain:
    if DATASET == 'opportunity':
        path_model = './convlstm_opportunity.hdf5'
    elif DATASET == 'dg':
        path_model = './convlstm_dg.hdf5'
    elif DATASET == 'wetlab':
        path_model = './convlstm_wetlab.hdf5'
    elif DATASET == '50salads':
        path_model = './convlstm_50salads.hdf5'

    model = keras.models.load_model(path_model)
    print ('loaded model from ... {}'.format(path_model))

from evaluate import evaluate
evaluate(model, 
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    NUM_CLASSES)