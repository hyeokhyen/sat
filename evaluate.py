import sklearn.metrics as metrics
from wilsonscore import wilsonscore
import numpy as np

def evaluate(model, 
    X_train_feat, y_train,
    X_val_feat, y_val,
    X_test_feat, y_test,
    NUM_CLASSES):
    X_train_prob = model.predict(X_train_feat)
    X_val_prob = model.predict(X_val_feat)
    X_test_prob = model.predict(X_test_feat)

    if len(X_train_prob.shape) > 1:
        assert len(X_train_prob.shape) == 2
        assert len(X_val_prob.shape) == 2
        assert len(X_test_prob.shape) == 2
        X_train_pred = np.argmax(X_train_prob, axis=1)
        X_val_pred = np.argmax(X_val_prob, axis=1)
        X_test_pred = np.argmax(X_test_prob, axis=1)
    else:
        X_train_pred = X_train_prob
        X_val_pred = X_val_prob
        X_test_pred = X_test_prob

    # f1 score
    if NUM_CLASSES == 2:
        train_mean_f1 = metrics.f1_score(y_train, X_train_pred, average='binary')
        val_mean_f1 = metrics.f1_score(y_val, X_val_pred, average='binary')
        test_mean_f1 = metrics.f1_score(y_test, X_test_pred, average='binary')

        train_weighted_f1 = -0.
        val_weighted_f1 = -.0
        test_weighted_f1 = -.0
    else:
        train_mean_f1 = metrics.f1_score(y_train, X_train_pred, average='macro')
        val_mean_f1 = metrics.f1_score(y_val, X_val_pred, average='macro')
        test_mean_f1 = metrics.f1_score(y_test, X_test_pred, average='macro')

        train_weighted_f1 = metrics.f1_score(y_train, X_train_pred, average='weighted')
        val_weighted_f1 = metrics.f1_score(y_val, X_val_pred, average='weighted')
        test_weighted_f1 = metrics.f1_score(y_test, X_test_pred, average='weighted')

    # wilson score
    wilson_score_low, wilson_score_high = wilsonscore(
        test_mean_f1, y_test.shape[0])
    print ('train: mean_f1: {:.4} | weighted_f1: {:.4}'.format(
        train_mean_f1, train_weighted_f1))
    print ('val: mean_f1: {:.4} | weighted_f1: {:.4}'.format(
        val_mean_f1, val_weighted_f1))
    print ('test: mean_f1: {:.4} | weighted_f1: {:.4}'.format(
        test_mean_f1, test_weighted_f1))
    print ('wilson score | low: {:.4} | high: {:.4}'.format(wilson_score_low, wilson_score_high))