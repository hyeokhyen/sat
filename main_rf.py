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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import copy
import keras

#----------------------------------------------------------
DATASET = '50salads' # opportunity, dg, wetlab, 50salads
#----------------------------------------------------------

print ('=== RF-SAT for {} ==='.format(DATASET))

from dataloader import dataloader
X_train, X_val, X_test, y_train, y_val, y_test, NUM_CLASSES, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, normal_loc, NB_SENSOR_CHANNELS = dataloader(DATASET)

# prepare dataset
from sliding_window import sliding_window
X_train = sliding_window(X_train, 
    (SLIDING_WINDOW_LENGTH, X_train.shape[1]),
    (SLIDING_WINDOW_STEP, 1))
X_train = X_train.astype(np.float32)

X_val = sliding_window(X_val, 
    (SLIDING_WINDOW_LENGTH, X_val.shape[1]),
    (SLIDING_WINDOW_STEP, 1))
X_val = X_val.astype(np.float32)

X_test = sliding_window(X_test, 
    (SLIDING_WINDOW_LENGTH, X_test.shape[1]),
    (SLIDING_WINDOW_STEP, 1))
X_test = X_test.astype(np.float32)

y_train_original = copy.deepcopy(y_train)
y_train_original = keras.utils.to_categorical(y_train_original, NUM_CLASSES)
y_train_original = np.asarray([[i[-1]] 
    for i in sliding_window(y_train_original, 
    (SLIDING_WINDOW_LENGTH, y_train_original.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_train_original.shape[1]))

y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_val = np.asarray([[i[-1]] 
    for i in sliding_window(y_val, 
    (SLIDING_WINDOW_LENGTH, y_val.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_val.shape[1]))

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_test = np.asarray([[i[-1]] 
    for i in sliding_window(y_test, 
    (SLIDING_WINDOW_LENGTH, y_test.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_test.shape[1]))

print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train_original.shape))
print(" ..after sliding window (val): inputs {0}, targets {1}".format(X_val.shape, y_val.shape))
print(" ..after sliding window (test): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

from ecdfRep import ecdfRep
_feat = ecdfRep(X_train[0], 15)
X_train_feat = np.empty((X_train.shape[0], _feat.shape[0]))
X_train_feat[:] = np.nan
for i in range(X_train.shape[0]):
    X_train_feat[i] = ecdfRep(X_train[i], 15)
assert np.all(np.isfinite(X_train_feat))

X_val_feat = np.empty((X_val.shape[0], _feat.shape[0]))
X_val_feat[:] = np.nan
for i in range(X_val.shape[0]):
    X_val_feat[i] = ecdfRep(X_val[i], 15)
assert np.all(np.isfinite(X_val_feat))

X_test_feat = np.empty((X_test.shape[0], _feat.shape[0]))
X_test_feat[:] = np.nan
for i in range(X_test.shape[0]):
    X_test_feat[i] = ecdfRep(X_test[i], 15)
assert np.all(np.isfinite(X_test_feat))

# normalize
X_mean = np.mean(X_train_feat, axis=0).reshape(
    (1,X_train_feat.shape[1])) 
X_std = np.std(X_train_feat, axis=0).reshape(
    (1,X_train_feat.shape[1]))
X_train_feat -= X_mean
X_train_feat /= X_std
X_val_feat -= X_mean
X_val_feat /= X_std
X_test_feat -= X_mean
X_test_feat /= X_std

print(" ..after feature extraction (training): inputs {0}, targets {1}".format(X_train_feat.shape, y_train_original.shape))
print(" ..after feature extraction (val): inputs {0}, targets {1}".format(X_val_feat.shape, y_val.shape))
print(" ..after feature extraction (test): inputs {0}, targets {1}".format(X_test_feat.shape, y_test.shape))

# get SAT
from label_augment_tool import overlap_jitter, transition_jitter

lam = 0.5
y_train_ov = overlap_jitter(y_train, NUM_CLASSES=NUM_CLASSES)
y_train_tr = transition_jitter(y_train, NUM_CLASSES=NUM_CLASSES)
y_train_jitter = lam*y_train_tr + (1-lam)*y_train_ov
y_train_jitter /= np.sum(y_train_jitter, axis=1).reshape((-1, 1))

y_train_jitter = np.asarray([[i[-1]] 
    for i in sliding_window(y_train_jitter, 
    (SLIDING_WINDOW_LENGTH, y_train_jitter.shape[1]),
    (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_train_jitter.shape[1]))
print ('label jitter {} | sebsequent activity interval context ({}) + own activity sequence context({}) ---'.format(y_train_jitter.shape, 1-lam, lam))

# check soft labels
idx_samples = np.where(np.all(y_train_jitter < 1., axis=1))[0]
_y_train_jitter = copy.deepcopy(y_train_jitter)

n_samples = 10
print ('---- {} samples data OR label ----'.format(n_samples))
from scipy.stats import multinomial
total_samples = n_samples * X_train_feat.shape[0]
X_train_samples = np.empty((total_samples, X_train_feat.shape[1]))
y_train_samples = np.empty((total_samples, y_train_jitter.shape[1]))
X_train_samples[:] = np.nan
y_train_samples[:] = np.nan
print ('prepare training label samples {} {}'.format(
    X_train_samples.shape, y_train_samples.shape))

for i_sample in range(n_samples):

    if i_sample == 0:
        X_train_samples[i_sample*X_train_feat.shape[0]:(i_sample+1)*X_train_feat.shape[0]] = X_train_feat
        y_train_samples[i_sample*y_train_jitter.shape[0]:(i_sample+1)*y_train_jitter.shape[0]] = y_train_original
        print ('{:03d}th copy original label for first sample ...'.format(i_sample+1))
        continue
    
    noises = np.random.normal(loc=normal_loc, scale=1.,
        size=X_train_feat.shape)
    X_train_samples[i_sample*X_train_feat.shape[0]:(i_sample+1)*X_train_feat.shape[0]] = X_train_feat + 0.001*noises
    #print ('data augment')

    _y_soft_samples = np.empty((idx_samples.shape[0], y_train_jitter.shape[1]))
    _y_soft_samples[:] = np.nan 
    for i, i_d in enumerate(idx_samples):
        y_prob = y_train_jitter[i_d].astype(np.float64)
        y_prob /= np.sum(y_prob)
        if np.sum(y_prob) > 1: # due to numerical precision
            y_prob += np.finfo(float).eps
            y_prob /= np.sum(y_prob)
        _y_soft_samples[i] = multinomial.rvs(1, y_prob, 1)
    assert np.all(np.isfinite(_y_soft_samples))
    assert not np.any((_y_soft_samples < 1) & (_y_soft_samples > 0))

    _y_train_jitter[idx_samples] = _y_soft_samples
    y_train_samples[i_sample*y_train_jitter.shape[0]:(i_sample+1)*y_train_jitter.shape[0]] = _y_train_jitter
    #print ('label jitter')
    print ('{:03d}th sampling ...'.format(i_sample+1))

y_train_samples[y_train_samples < 0.5] = 0.
y_train_samples[y_train_samples > 0.5] = 1.
assert np.all(np.isfinite(X_train_samples))
assert np.all(np.isfinite(y_train_samples))
assert not np.any((y_train_samples < 1) & (y_train_samples > 0)), '{}\n{}\n{}, {}'.format(
    np.where(np.any((y_train_samples < 1) & (y_train_samples > 0), axis=1))[0],
    y_train_samples[np.where(np.any((y_train_samples < 1) & (y_train_samples > 0), axis=1))[0]],
    np.amax(y_train_samples[np.where(np.any((y_train_samples < 1) & (y_train_samples > 0), axis=1))[0]]),
    np.amin(y_train_samples[np.where(np.any((y_train_samples < 1) & (y_train_samples > 0), axis=1))[0]]))
print(" .. after sampling (train): inputs {0}, targets {1}".format(
    X_train_samples.shape, y_train_samples.shape))

if y_train_samples.shape[1] == 2:
    y_train_samples = np.argmax(y_train_samples, axis=1)

# model definition
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=10,
    min_samples_leaf=5,
    n_jobs=-1,
    verbose=0)    

model.fit(X_train_samples, y_train_samples)

# evaluate
y_train = np.argmax(y_train_original, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)

from evaluate import evaluate
evaluate(model, 
    X_train_feat, y_train,
    X_val_feat, y_val,
    X_test_feat, y_test,
    NUM_CLASSES)