import keras
import numpy as np
from scipy.signal import windows as sciwin
from scipy.ndimage import convolve1d
import time

def get_activity_sequences(y):
    # length of each label session in consecutive order
    y_sess = []
    y_len_sess = []
    y_len_time = []

    len_sess = 1
    y_prev = y[0]
    num_y = len(y)
    for i_l in range(1, num_y):
        label = y[i_l]

        if label != y_prev or i_l == num_y-1:
            y_sess.append(y_prev)
            y_len_sess.append(len_sess)
            y_len_time.append(i_l)
            # renew
            len_sess = 1
            y_prev = label
        else:
            len_sess += 1    
    y_sess = np.array(y_sess)
    y_len_sess = np.array(y_len_sess)
    y_len_time = np.array(y_len_time)

    return y_sess, y_len_sess, y_len_time    

def transition_jitter(y_train,
    NUM_CLASSES=None,
    boundary_ub=None,
    boundary_ub_pct=None,
    boundary_lb_pct=None,
    beta=None):
    print ('--- activity transition context ---')
    kernel_window = getattr(sciwin, 'gaussian')
    
    # transitional activity
    y_categorical = keras.utils.to_categorical(y_train, NUM_CLASSES)

    # 1) find the change of label points
    y_activity, y_activity_len, y_activity_time = get_activity_sequences(y_train)
    #print (y_activity)
    #print (y_activity_len)
    #print (y_activity_time)

    # 2) find boundary_limit
    if boundary_ub is None: # 50%
        if boundary_ub_pct is None:
            boundary_ub_pct = 50
        boundary_ub = np.percentile(y_activity_len[y_activity == 0],
            boundary_ub_pct)
    #print ('boundary_ub_pct', boundary_ub_pct)
    #print ('boundary_ub', boundary_ub)

    if boundary_lb_pct is None:
        boundary_lb_pct = 0.1*boundary_ub_pct
    #print ('boundary_lb_pct', boundary_lb_pct)

    # 3) define alpha control parameter
    if beta is None:
        beta = boundary_ub**2
    #print ('beta', beta)    

    # 4) find boundary_length
    y_activity_len[y_activity_len == 0] = 1 # prevent division by 0
    smoothing_scale = beta/y_activity_len

    act_times = np.where(y_activity != 0)[0]
    #print ('number of activities:', act_times.shape[0])

    start_time = time.time()    
    y_processed = np.zeros(y_categorical.shape)
    for i in act_times:
        scale = int(np.clip(smoothing_scale[i], 
            y_activity_len[i]*boundary_lb_pct, 
            y_activity_len[i]*boundary_ub_pct))
        if scale < 5:
            scale = 5

        # window size
        filter_window = kernel_window(scale, 0.1*scale)
        filter_window /= np.sum(filter_window)

        # start boundary
        time_start = y_activity_time[i-1] - scale
        if time_start < 0:
            time_start = 0
        time_end = y_activity_time[i-1] + scale
        if time_end > len(y_train)-1:
            time_end = len(y_train)-1
        y_window = y_categorical[time_start:time_end,:].T
        #print ('start', scale, y_window.shape)

        y_window_smoothe = convolve1d(y_window, filter_window,
            axis=1, mode='constant', cval=0.)
        y_window_smoothe /= np.sum(y_window_smoothe, axis=0).reshape(1,-1)
        y_window_smoothe = np.clip(y_window_smoothe, 0, 1)

        y_processed[time_start:time_end,:] += y_window_smoothe.T

        # end boundary
        time_start = y_activity_time[i] - scale
        if time_start < 0:
            time_start = 0
        time_end = y_activity_time[i] + scale
        if time_end > len(y_train)-1:
            time_end = len(y_train)-1
        y_window = y_categorical[time_start:time_end,:].T
        #print ('end', scale, y_window.shape)

        y_window_smoothe = convolve1d(y_window, filter_window,
            axis=1, mode='constant', cval=0.)
        y_window_smoothe /= np.sum(y_window_smoothe, axis=0).reshape(1,-1)
        y_window_smoothe = np.clip(y_window_smoothe, 0, 1)

        y_processed[time_start:time_end,:] += y_window_smoothe.T
        #print ('{}'.format(y_categorical[time_start:time_end,:]))

    end_time = time.time()
    #print ('elapsed filter time: {:.2f}s'.format(end_time-start_time))
    
    # fill the zeros which are ones
    i_zero_y_processed = np.where(np.all(y_processed==0, axis=1))[0]
    #print (i_zero_y_processed)
    #print (i_zero_y_processed.shape, y_processed.shape)
    y_processed[i_zero_y_processed] = y_categorical[i_zero_y_processed]
    y_processed /= np.sum(y_processed, axis=1).reshape((-1, 1))
    assert np.all(np.abs(np.sum(y_processed, axis=1) - 1.) < 1e-5), 'label probability does not sum up to 1\n{}\n{}'.format(np.sum(y_processed, axis=1), y_processed)

    return y_processed

def overlap_jitter(y_train, 
    NUM_CLASSES=None,
    boundary_ub=None,
    boundary_ub_pct=None,
    boundary_lb=None,
    boundary_lb_pct=None,
    alpha=None):
    kernel_window = getattr(sciwin, 'gaussian')
    print ('--- subsequent activity overlap context ---')

    # transitional activity
    y_categorical = keras.utils.to_categorical(y_train, NUM_CLASSES)

    # 1) find the change of label points
    y_activity, y_activity_len, y_activity_time = get_activity_sequences(y_train)

    # 2) find boundary_limit
    if boundary_ub is None: # 50%
        if boundary_ub_pct is None:
            boundary_ub_pct = 50
        boundary_ub = np.percentile(y_activity_len[y_activity == 0],
            boundary_ub_pct)
    #print ('boundary_ub_pct', boundary_ub_pct)
    #print ('boundary_ub', boundary_ub)

    if boundary_lb is None: # 25%
        if boundary_lb_pct is None:
            boundary_lb_pct = 0.1*boundary_ub_pct
        boundary_lb = np.percentile(y_activity_len[y_activity == 0],
            boundary_lb_pct)
    #print ('boundary_lb_pct', boundary_lb_pct)
    #print ('boundary_lb', boundary_lb)

    # 3) define alpha control parameter
    if alpha is None:
        alpha = boundary_ub**2
    #print ('alpha', alpha)    

    # 4) find boundary_length
    y_activity_len[y_activity_len == 0] = 1 # prevent division by 0
    smoothing_scale = alpha/y_activity_len
    #print (smoothing_scale)
    #print (np.max(smoothing_scale), np.min(smoothing_scale))
    
    smoothing_scale[smoothing_scale > boundary_ub] = boundary_ub 
    smoothing_scale[smoothing_scale < boundary_lb] = boundary_lb
    #print (smoothing_scale)
    #print ('smoothing_scale:', np.max(smoothing_scale), np.min(smoothing_scale))

    # for minimum window size
    smoothing_scale[smoothing_scale < 5] = 5
    #print ('smoothing_scale correcteed:', np.max(smoothing_scale), np.min(smoothing_scale))
        
    # 6) smoothe
    null_times = np.where(y_activity == 0)[0]
    #print (null_times)
    #print (y_activity_time[null_times])

    y_processed = np.zeros(y_categorical.shape)
    for i in null_times[:-1]:    
        scale = int(smoothing_scale[i])

        filter_window = kernel_window(scale, 0.1*scale)
        filter_window /= np.sum(filter_window)

        # start segment
        time_start = y_activity_time[i-1] - scale
        if time_start < 0:
            time_start = 0
        time_end = y_activity_time[i-1] + scale
        if time_end > len(y_train)-1:
            time_end = len(y_train)-1
        y_window = y_categorical[time_start:time_end,:].T

        y_window_smoothe = convolve1d(y_window, filter_window,
            axis=1, mode='constant', cval=0.)
        y_window_smoothe /= np.sum(y_window_smoothe, axis=0).reshape(1,-1)
        y_window_smoothe = np.clip(y_window_smoothe, 0, 1)

        y_processed[time_start:time_end,:] += y_window_smoothe.T

        # end boundary
        time_start = y_activity_time[i] - scale
        if time_start < 0:
            time_start = 0
        time_end = y_activity_time[i] + scale
        if time_end > len(y_train)-1:
            time_end = len(y_train)-1
        y_window = y_categorical[time_start:time_end,:].T
        #print ('end', scale, y_window.shape)

        y_window_smoothe = convolve1d(y_window, filter_window,
            axis=1, mode='constant', cval=0.)
        y_window_smoothe /= np.sum(y_window_smoothe, axis=0).reshape(1,-1)
        y_window_smoothe = np.clip(y_window_smoothe, 0, 1)

        y_processed[time_start:time_end,:] += y_window_smoothe.T
        #print ('{}'.format(y_categorical[time_start:time_end,:]))

    # fill the zeros which are ones
    i_zero_y_processed = np.where(np.all(y_processed==0, axis=1))[0]
    y_processed[i_zero_y_processed] = y_categorical[i_zero_y_processed]
    y_processed /= np.sum(y_processed, axis=1).reshape((-1, 1))
    assert np.all(np.abs(np.sum(y_processed, axis=1) - 1.) < 1e-5), 'label probability does not sum up to 1\n{}\n{}'.format(np.sum(y_processed, axis=1), y_processed)

    return y_processed