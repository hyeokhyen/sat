import keras
import numpy as np
from sliding_window import sliding_window
from label_augment_tool import overlap_jitter, transition_jitter
from scipy.stats import multinomial
import copy

class batch_generator(keras.utils.Sequence):

    def __init__(self, X_train, y_train,
        NUM_CLASSES=18,
        batch_size=128,
        SLIDING_WINDOW_LENGTH=24,
        SLIDING_WINDOW_STEP=12,
        NB_SENSOR_CHANNELS=113,
        is_validate=False,
        shuffle=True,
        # overlap jitter
        ov_boundary_ub=None,
        ov_boundary_ub_pct=None,
        ov_boundary_lb=None,
        ov_boundary_lb_pct=None,
        ov_alpha=None,
        # transition jitter
        tr_boundary_ub=None,
        tr_boundary_ub_pct=None,
        tr_boundary_lb_pct=None,
        tr_beta=None,
        # combined
        lam=0.5,
        # training batch generation
        # label
        is_soft_label=True,
        is_sample_label_dist=True,
        is_mix_label_original=True,
        # input
        is_data_augment=True,
        data_augment_noise_type='normal',
        normal_loc=0.):

        if is_validate:
            y_categorical = keras.utils.to_categorical(y_train, NUM_CLASSES)
        else:
            list_label_case = []
            if is_soft_label:
                list_label_case.append('soft_label')
            if is_sample_label_dist:
                list_label_case.append('sample_label_dist')
            if is_mix_label_original:
                list_label_case.append('mix_label_original')
                y_categorical_original = keras.utils.to_categorical(y_train, NUM_CLASSES)
                self.y_original = np.asarray([[i[-1]] 
                    for i in sliding_window(y_categorical_original, 
                    (SLIDING_WINDOW_LENGTH, y_categorical_original.shape[1]),
                    (SLIDING_WINDOW_STEP, 1))]).reshape(
                        (-1, y_categorical_original.shape[1]))

            self.is_soft_label = is_soft_label
            self.is_sample_label_dist = is_sample_label_dist
            self.is_mix_label_original = is_mix_label_original
            self.list_label_case = list_label_case
            
            # SAT
            y_ov = overlap_jitter(y_train,
                NUM_CLASSES=NUM_CLASSES,
                boundary_ub=ov_boundary_ub, 
                boundary_ub_pct=ov_boundary_ub_pct, 
                boundary_lb=ov_boundary_lb, 
                boundary_lb_pct=ov_boundary_lb_pct, 
                alpha=ov_alpha)
            
            y_tr = transition_jitter(y_train,
                NUM_CLASSES=NUM_CLASSES,
                boundary_ub=tr_boundary_ub,
                boundary_ub_pct=tr_boundary_ub_pct,
                boundary_lb_pct=tr_boundary_lb_pct,
                beta=tr_beta)

            print ('--- sebsequent activity interval context ({}) + own activity sequence context({}) ---'.format(1-lam, lam))
            y_categorical = lam*y_tr + (1-lam)*y_ov        
            y_categorical = y_categorical.astype(np.float64)
            y_categorical /= np.sum(y_categorical, axis=1).reshape((-1,1))
        
        # Sensor data is segmented using a sliding window mechanism
        X_train = sliding_window(X_train, 
            (SLIDING_WINDOW_LENGTH, X_train.shape[1]),
            (SLIDING_WINDOW_STEP, 1))
        X_train = X_train.astype(np.float32)
        self.X_train = X_train.reshape(-1, 
            SLIDING_WINDOW_LENGTH, 
            NB_SENSOR_CHANNELS, 
            1)

        self.y_train = np.asarray([[i[-1]] 
            for i in sliding_window(y_categorical, 
            (SLIDING_WINDOW_LENGTH, y_categorical.shape[1]),
            (SLIDING_WINDOW_STEP, 1))]).reshape((-1, y_categorical.shape[1]))
        #print("... after sliding window (training): inputs {0}, targets {1}".format(self.X_train.shape, self.y_train.shape))
        
        self.batch_size = batch_size
        self.indexes = np.arange(self.X_train.shape[0])
        self.NUM_CLASSES = NUM_CLASSES
        self.is_validate = is_validate

        self.is_data_augment = is_data_augment
        self.data_augment_noise_type = data_augment_noise_type
        self.normal_loc = normal_loc

        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(
            np.floor(
                self.X_train.shape[0]/self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle == True and not self.is_validate:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X = self.X_train[indexes,:,:,:]
        if self.is_data_augment and not self.is_validate:
            if self.data_augment_noise_type == 'normal':
                noises = np.random.normal(loc=self.normal_loc, scale=1., size=X.shape)
            elif self.data_augment_noise_type == 'uniform':
                noises = np.random.uniform(low=-1., high=1., size=X.shape)
            X += 0.001*noises

        if self.is_validate:
            y = self.y_train[indexes,:]
        else:
            num_label_case = np.sum([
                self.is_soft_label, 
                self.is_sample_label_dist,
                self.is_mix_label_original])
            assert num_label_case == len(self.list_label_case)

            idx_label_case = np.random.randint(num_label_case, size=len(indexes))
            y = np.zeros((len(indexes), self.y_train.shape[1]))
            for i, ind in enumerate(indexes):
                name_label_case = self.list_label_case[idx_label_case[i]]
                if name_label_case == 'soft_label':
                    y[i] = self.y_train[ind]
                elif name_label_case == 'sample_label_dist':
                    y_prob = self.y_train[ind].astype(np.float64)
                    y_prob /= np.sum(y_prob)
                    if np.sum(y_prob) > 1: # due to numerical precision
                        y_prob += np.finfo(float).eps
                        y_prob /= np.sum(y_prob)
                    y[i] = multinomial.rvs(1, y_prob, 1)
                elif name_label_case == 'mix_label_original':
                    y[i] = self.y_original[ind]
                else:
                    raise ValueError('not existing label case: {}'.format(name_label_case))
                #print (i, ind, name_label_case,y[i])

        assert np.sum(y) == len(indexes)
        assert np.all(np.abs(np.sum(y, axis=1) - 1.) < 1e-5), 'label probability does not sum up to 1\n{}\n{}'.format(np.sum(y, axis=1), y)        

        return X, y

if __name__ == '__main__':
    pass

