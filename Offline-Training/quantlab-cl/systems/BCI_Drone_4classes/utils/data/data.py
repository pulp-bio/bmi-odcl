# 
# data.py
# 
# Author(s):
# Lan Mei <lanmei@student.ethz.ch>
#
# Copyright (c) 2024 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# 
""" Load EEG data for MI-BMINet. """

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch as t
import torch.nn.functional as F
import json
import torch.utils.data as tud
import random
import numpy as np
from torchvision.transforms import Compose
from os import path, walk
import pandas as pd
from scipy import signal

import os

from systems.utils.data import default_dataset_cv_split

import pickle
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window


# filters
def notch_multiEEG(data,fs):
    '''
    50Hz multi channel EEG
    
    Params
    ------
    data: np array (n_ch,n_s)
        EEG data
    fs: float
        sampling frequency
    
    Return
    ------
    data_filt: np array (n_ch, n_s)
        filtered EEG data 
    '''
    
    w0 = 50/(fs/2)
    Q = 30
    notch_filter_50Hz = signal.iirnotch(w0, Q)

    
    data_filt = np.zeros(data.shape)
    for chan in range(data.shape[0]):
         
       data_filt[chan, :]= signal.lfilter(*notch_filter_50Hz, data[chan, :])
        
    return data_filt

    
def bandpass_multiEEG(data, fs, f_low, f_high, order):
    '''
    Bandpass multi channel EEG
    
    Params
    ------
    data: np array (n_ch,n_s)
        EEG data
    f_low: float
        lower corner frequency [Hz]
    f_high: float
        upper corner_frequency [Hz]
    fs: float
        sampling frequency
    
    Return
    ------
    data_filt: np array (n_ch, n_s)
        filtered EEG data 
    '''

    nyq = 0.5 * fs
    low_freq = f_low / nyq
    high_freq = f_high / nyq
    filt_coef = signal.butter(N=order, Wn=[low_freq, high_freq], btype='bandpass', output='sos')
    
    for chan in range(data.shape[0]):
        data[chan, :] = signal.sosfilt(filt_coef, data[chan, :])
    return data


# functions to load and preprocess trial
def get_data(directory, n_files, fs, filter_config, ds=1):

    intertrial_sec = 4  # at least 4 seconds between trials
    samples_stimulus = 1950 # 1950  # stimulus is 2000 samples, but some are lost due to bug in BioWolf app (4s?)
    offset_sec = -5  # startpoint for the stimulus to load. choose Timewindo (0,samples_sec) for classification in config file: t1_factor, t2_factor
    offset = int(fs * offset_sec)  # number of samples how much later than the stimulus a trial starts
    samples = samples_stimulus - offset  # number of samples of trial before cut
    intertrial = int(fs * intertrial_sec)  # number of samples in between trials
    number_of_trials = 20 # BCI_Drone_4classes: 20 for 2-classes data, 40 for 4-classes data

    # trigger values
    trigger_left = 50
    trigger_right = 60
    trigger_tongue = 70
    trigger_nothing = 100

    # labels for classifier
    lb_nothing = 0
    lb_left = 1
    lb_right = 2
    lb_tongue = 3

    N_ch = 8

    number_of_runs = n_files

    file_index = 0  # file index
    runs = np.zeros((number_of_runs, number_of_trials, N_ch, samples), dtype=np.float32)
    y_runs = np.zeros((number_of_runs, number_of_trials), dtype=np.float32)

    count_trials = 0

    preprocess_mean = 0
    for root, dirs, files in walk(directory):
        files.sort()
        for file in files:
            print(file)
            trials = np.zeros((number_of_trials, N_ch, samples), dtype=np.float32)
            y = np.zeros(number_of_trials, dtype=np.float32)
            file_path = path.join(root, file)
            df = pd.read_csv(file_path, sep=',')

            data = df.to_numpy(dtype=np.float32)
            X = np.transpose(data[:, :])  # X: [channels+triggers, samples]

            temp_afterfilt_X = X[:-1, :].copy()
            temp_afterfilt_X = notch_multiEEG(temp_afterfilt_X, fs)
            temp_afterfilt_X = bandpass_multiEEG(temp_afterfilt_X, fs, filter_config['fc_low'], filter_config['fc_high'],
                                              filter_config['order'])

            
            # Subtracting moving avg.
            N = int(250/2)
            for i in range(N_ch):
                temp_afterfilt_X_moving_avg = np.convolve(temp_afterfilt_X[i], np.ones(N)/N, mode='same')
                temp_afterfilt_X[i] = temp_afterfilt_X[i] - temp_afterfilt_X_moving_avg
            
            '''
            # Only if we use ASR for artifact removal.
            # ASR
            asr = ASR(method="euclid", sfreq=500) 
            train_idx = np.arange(240 * fs, 270 * fs, dtype=int) # 10-40: 2.3; 200-260: 0.9; 240-260(270): 0.6637354493141174
            #240-280: 0.9324485659599304
            _, sample_mask = asr.fit(temp_afterfilt_X[:, train_idx])

            # Apply filter using sliding (non-overlapping) windows
            X_tmp = sliding_window(temp_afterfilt_X, window=int(fs), step=int(fs))
            Y_tmp = np.zeros_like(X_tmp)
            for i in range(X_tmp.shape[1]):
                Y_tmp[:, i, :] = asr.transform(X_tmp[:, i, :])

            raw = X_tmp.reshape(N_ch, -1)  # reshape to (n_chans, n_times)
            temp_afterfilt_X = Y_tmp.reshape(N_ch, -1)

            '''

            n_samples = X.shape[1]  

            k = 0  # trial
            sample = 0

            cur_sample_count = 0
            cur_sample_len = 0

            while sample < n_samples:
                    
                # ===================== Dataset A, left / right MM - START =====================
                if X[-1, sample] == trigger_left:

                    start_trial = sample + offset  # start trial is the sample the stimulus is detected plus the offset
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_left  # put label of left
                    y[k] = y[k] - 1  # two classes: left:0

                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    ## Only if we downsample the data
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y

                elif X[-1, sample] == trigger_right: 

                    start_trial = sample + offset  # start trial used for classification
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_right  # put label of right
                    y[k] = y[k] - 1  # two classes: right:1

                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    ## Only if we downsample the data
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y

                # ===================== Dataset A, left / right MM -  END  =====================

                '''
                # ===================== Dataset B, tongue MM / rest - START =====================
                if X[-1, sample] == trigger_nothing: 

                    start_trial = sample + offset  # start trial is the sample the stimulus is detected plus the offset
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_left  # put label of left
                    y[k] = y[k] - 1  # two classes: left:0

                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    ## Only if we downsample the data
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y

                elif X[-1, sample] == trigger_tongue: 

                    start_trial = sample + offset  # start trial used for classification
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_right  # put label of right
                    y[k] = y[k] - 1  # two classes: right:1

                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    ## Only if we downsample the data
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y
                # ===================== Dataset B, tongue MM / rest -  END  =====================
                '''

                '''
                # ===================== Dataset B, left / right / tongue MM / rest - START =====================
                if X[-1, sample] == trigger_left: 

                    start_trial = sample + offset  # start trial is the sample the stimulus is detected plus the offset
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_left  
                    y[k] = y[k] - 1  

                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    ## Only if we downsample the data
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y

                elif X[-1, sample] == trigger_right: 

                    start_trial = sample + offset  # start trial used for classification
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_right  
                    y[k] = y[k] - 1  

                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    ## Only if we downsample the data
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y

                elif X[-1, sample] == trigger_nothing:  

                    start_trial = sample + offset  # start trial used for classification
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_nothing  
                    y[k] = y[k] + 3  
                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1

                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y
                    
                elif X[-1, sample] == trigger_tongue:  

                    start_trial = sample + offset  # start trial used for classification
                    stop_trial = start_trial + samples  # stop trial
                    temp_filt = temp_afterfilt_X[0:N_ch, start_trial:stop_trial]  # 8 channels modify

                    trials[k] = temp_filt
                    y[k] = lb_tongue  
                    y[k] = y[k] - 1  
                    sample = sample + samples_stimulus + intertrial  # skip to the next sample
                    k += 1  # go to the next trial
                    count_trials += 1
                    
                    # if ds > 1:
                    #     samples_ds = int(np.ceil(samples / ds))
                    #     X_ds = np.zeros((N_ch, samples_ds), dtype=np.float64)
                    #     for chan in range(N_ch):
                    #         X_ds[chan] = signal.decimate(trial[chan, :], ds)
                    #     trial = X_ds
                    # return trial, y

                # ===================== Dataset B, left / right / tongue MM / rest -  END  =====================
                '''

                else:
                    sample += 1  # go to the next sample


            runs[file_index] = trials
            y_runs[file_index] = y
            file_index += 1

    X = runs.reshape(number_of_runs * number_of_trials, N_ch, samples)
    y = y_runs.reshape(number_of_runs * number_of_trials)


    if ds > 1:
        samples_ds = int(np.ceil(samples / ds))
        X_ds = np.zeros((number_of_runs * number_of_trials, N_ch, samples_ds), dtype=np.float32)
        for trial in range(number_of_runs * number_of_trials):
            for chan in range(N_ch):
                X_ds[trial, chan] = signal.decimate(X[trial, chan, :], ds)
        X = X_ds

    #print(count_trials)

    X_new_shape = []
    y_new_shape = []

    for trial_idx in range(number_of_runs * number_of_trials):
        if np.any(X[trial_idx]):
            # Filter out artifacts.
            max_val = np.max(np.abs(X[trial_idx, :, 2550:4450]))
            X_new_shape.append(X[trial_idx])
            y_new_shape.append(y[trial_idx])

            '''
            # Only used if a threshold is set.
            if max_val < 120:
                X_new_shape.append(X[trial_idx])
                y_new_shape.append(y[trial_idx])
            '''


    X = np.array(X_new_shape)
    y = np.array(y_new_shape)

    print(X.shape)

    return X,y


class BCI_Drone_Dataset(t.utils.data.Dataset):

    def __init__(self, root, fs=500, transform=None, filter_config=None, ds=1):
        self.root = root
        self.fs = fs
        self.transform = transform
        self.filter_config = filter_config
        self.ds = ds
        self.samples, self.labels = self._load_data()

    def __len__(self):  # needed for for-loop in rooms.py
        return len(self.samples)

    def __getitem__(self, idx):  # gets called by for-loop in rooms.py

        sample = self.samples[idx, :, :]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def _load_data(self):
        n_files = len(os.listdir(self.root))
        X, y = get_data(self.root, n_files, self.fs, self.filter_config, self.ds)

        data_return = t.Tensor(X).to(dtype=t.float)
        class_return = t.Tensor(y).to(dtype=t.long)

        return data_return, class_return


class BCI_Drone_Dataset_With_Buffer(t.utils.data.Dataset):
    # Data loader for ER with buffer
    def __init__(self, root, bufroot, fs=500, transform=None, filter_config=None, ds=1, X_buf=None, y_buf=None, buffer_size=20, seen_examples=0, phase=1):
        self.root = root
        self.bufroot = bufroot
        self.fs = fs
        self.transform = transform
        self.filter_config = filter_config
        self.ds = ds
        self.X_buf = X_buf
        self.y_buf = y_buf
        self.buffer_size = buffer_size
        self.seen_examples = seen_examples
        self.phase = phase
        self.samples, self.labels = self._load_data()

    def __len__(self):  # needed for for-loop in rooms.py
        return len(self.samples)

    def __getitem__(self, idx):  # gets called by for-loop in rooms.py

        sample = self.samples[idx, :, :]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def _load_data(self):
        print("Loading dataset with Buffer... For ER.")
        n_files = len(os.listdir(self.root))
        X, y = get_data(self.root, n_files, self.fs, self.filter_config, self.ds)

        initialize = False
        if self.X_buf is None or self.y_buf is None: # initialization
            print("Initializing buffer... ")
            self.X_buf = np.zeros((self.buffer_size, X.shape[1], X.shape[2]), dtype=np.float32)
            self.y_buf = np.zeros((self.buffer_size), dtype=np.float32)
            initialize = True

        if not initialize:
            concat_X_buf = self.X_buf
            concat_y_buf = self.y_buf
            if self.seen_examples < self.buffer_size:
                concat_X_buf = self.X_buf[:self.seen_examples]
                concat_y_buf = self.y_buf[:self.seen_examples]

            X = np.concatenate((X, concat_X_buf), axis=0)
            y = np.concatenate((y, concat_y_buf), axis=0)
            print("Buffered! Total shape: ")
            print(X.shape)
            #print(self.X_buf.shape)
        else: 
            print("Not yet buffered! Shape: ")
            print(X.shape)

        data_return = t.Tensor(X).to(dtype=t.float)
        class_return = t.Tensor(y).to(dtype=t.long)

        print("Updating Buffer... For the next ER experience.")

        # Start updating the buffer AFTER using the current buffer!
        n_files_buf = len(os.listdir(self.bufroot))
        X_buf_tmp, y_buf_tmp = get_data(self.bufroot, n_files_buf, self.fs, self.filter_config, self.ds)
        print("Seen examples: ")
        print(self.seen_examples)
        print("Obtained buffer session shape: ")
        print(X_buf_tmp.shape)
        prev_seen_samples = self.seen_examples
        for i in range(X_buf_tmp.shape[0]):
            if self.seen_examples < self.buffer_size:
                self.X_buf[prev_seen_samples+i] = X_buf_tmp[i]
                self.y_buf[prev_seen_samples+i] = y_buf_tmp[i]
                self.seen_examples += 1
            else:
                self.seen_examples += 1
                rand = np.random.randint(0, self.seen_examples)
                if rand < self.buffer_size:
                    self.X_buf[rand] = X_buf_tmp[i]
                    self.y_buf[rand] = y_buf_tmp[i]
        print("Seen examples After updating: ")
        print(self.seen_examples)
        print("Buffer shape: ")
        print(self.X_buf.shape)

        saved_buffer = {}
        saved_buffer['seen_examples'] = self.seen_examples
        saved_buffer['X_buf'] = self.X_buf
        saved_buffer['y_buf'] = self.y_buf
        with open('buffer_saved/saved_buffer_{}.pickle'.format(self.phase), 'wb') as handle:
            pickle.dump(saved_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return data_return, class_return

        
def load_data_set(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: Compose,
                  use_pseudo_folds : bool = False,
                  fs : int = 500,
                  filter_type : str = "bandpass",
                  fc_low : float = 0.5,
                  fc_high : float = 100.0,
                  order : int = 4,
                  ds : int = 1) -> torch.utils.data.Dataset:
                  
    filter_config = {
        "type": filter_type, 
        "fc_low": fc_low, 
        "fc_high": fc_high, 
        "order": order
    }

    print(filter_config)
                  
    if partition in {'train', 'valid', 'test', "full_sess", "full_sess_val", 'train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_6', 'train_7', 'valid_1', 'valid_2', 'valid_3', 'valid_4', 'valid_5', 'valid_6', 'valid_7'}:
        if n_folds > 1 and not use_pseudo_folds:
            if partition in {'train', 'valid', "full_sess", "full_sess_val"}: 
                # ============= N-fold Within-session Classification - Dataset A =============
                # This name can be modified to the folder name of the data session to be classified in 5-fold CV experiments
                dir_data = os.path.join(path_data, 'DatasetA_S1_1127') # 'DatasetA_S2_1130', 'DatasetA_S3_1205', ...
                '''
                # ============= N-fold Within-session Classification - Dataset B =============
                # This name can be modified to the folder name of the data session to be classified in 5-fold CV experiments
                dir_data = os.path.join(path_data, 'SubjectA_1129_S1') # 'SubjectA_1206_S2', 'SubjectA_1213_S3', ...
                '''
                print(f"Loading data from: {dir_data}")
                dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config = filter_config, ds = ds)
                train_fold_indices, valid_fold_indices = default_dataset_cv_split(dataset=dataset, n_folds=n_folds, current_fold_id=current_fold_id, cv_seed=cv_seed)
                
                if partition == 'train':
                    # CV 5-fold:
                    dataset = torch.utils.data.Subset(dataset, train_fold_indices)

                elif partition == 'valid':
                    # CV 5-fold:
                    dataset = torch.utils.data.Subset(dataset, valid_fold_indices)

            else: 
                # ============= N-repetition TL/LwF - Dataset B =============
                if partition in {'train_1', 'train_2', 'train_3', 'train_4', 'valid_1', 'valid_2', 'valid_3', 'valid_4'}:
                    if partition == "train_1":
                        dir_data = os.path.join(path_data, 'SubjectA_1129_S1')
                    elif partition == "valid_1":
                        dir_data = os.path.join(path_data, 'SubjectA_1129_S1_t3v2', 'SubjectA_1129_S1_test') 
                    elif partition == "train_2":
                        dir_data = os.path.join(path_data, 'SubjectA_1206_S2_t3v2', 'SubjectA_1206_S2_train')
                    elif partition == "valid_2":
                        dir_data = os.path.join(path_data, 'SubjectA_1206_S2_t3v2', 'SubjectA_1206_S2_test')
                    elif partition == "train_3":
                        dir_data = os.path.join(path_data, 'SubjectA_1213_S3_t3v2', 'SubjectA_1213_S3_train')
                    elif partition == "valid_3":
                        dir_data = os.path.join(path_data, 'SubjectA_1213_S3_t3v2', 'SubjectA_1213_S3_test')
                    elif partition == "train_4":
                        dir_data = os.path.join(path_data, 'SubjectA_1220_S4_t3v2', 'SubjectA_1220_S4_train')
                    elif partition == "valid_4":
                        dir_data = os.path.join(path_data, 'SubjectA_1220_S4_t3v2', 'SubjectA_1220_S4_test')
                    print(f"Loading data from: {dir_data}")
                    dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config=filter_config, ds=ds)

                else:
                # ============= N-repetition ER with buffer - Dataset B =============
                    if partition == "train_ER_1":
                        dir_data = os.path.join(path_data, 'SubjectA_1129_S1')
                    elif partition == "train_ER_2":
                        dir_data = os.path.join(path_data, 'SubjectA_1206_S2_t3v2', 'SubjectA_1206_S2_train')
                    elif partition == "train_ER_3":
                        dir_data = os.path.join(path_data, 'SubjectA_1213_S3_t3v2', 'SubjectA_1213_S3_train')
                    elif partition == "train_ER_4":
                        dir_data = os.path.join(path_data, 'SubjectA_1220_S4_t3v2', 'SubjectA_1220_S4_train')

                    if partition in {'train_ER_1', 'train_ER_2', 'train_ER_3', 'train_ER_4'}:
                        cur_ER_phase = int(partition[-1])
                        print("Current ER phase: \n".format(cur_ER_phase))
                        X_buf = None
                        y_buf = None
                        seen_examples = 0
                        buf_dir_data = dir_data

                        print(f"Loading buffer data from: {buf_dir_data}")
                        buffer_size = 20
                        if os.path.isfile('buffer_saved/saved_buffer_{}.pickle'.format(cur_ER_phase-1)):
                            with open('buffer_saved/saved_buffer_{}.pickle'.format(cur_ER_phase-1), 'rb') as handle:
                                saved_buffer = pickle.load(handle)
                            seen_examples = saved_buffer['seen_examples']
                            X_buf = saved_buffer['X_buf']
                            y_buf = saved_buffer['y_buf']
                        dataset = BCI_Drone_Dataset_With_Buffer(root=dir_data, bufroot=buf_dir_data, fs=fs, transform=transform, 
                        filter_config=filter_config, ds=ds, X_buf=X_buf, y_buf=y_buf, buffer_size=buffer_size, seen_examples=seen_examples, phase=cur_ER_phase)

                    else:
                        dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config=filter_config, ds=ds)


        else:
            # ============= Single Within-session Training - Dataset B =============
            if partition == "train" or partition == "train_ER":
                dir_data = os.path.join(path_data, 'SubjectA_1129_S1')
            elif partition == 'valid':
                dir_data = os.path.join(path_data, 'SubjectA_1129_S1') 
            elif partition == 'test':
                dir_data = os.path.join(path_data, 'SubjectA_1129_S1') 

            # ============= Single Run TL/LwF - Dataset B =============
            elif partition == "train_1":
                dir_data = os.path.join(path_data, 'SubjectA_1129_S1')
            elif partition == "valid_1":
                dir_data = os.path.join(path_data, 'SubjectA_1129_S1_t3v2', 'SubjectA_1129_S1_test') 
            elif partition == "train_2":
                dir_data = os.path.join(path_data, 'SubjectA_1206_S2_t3v2', 'SubjectA_1206_S2_train')
            elif partition == "valid_2":
                dir_data = os.path.join(path_data, 'SubjectA_1206_S2_t3v2', 'SubjectA_1206_S2_test')
            elif partition == "train_3":
                dir_data = os.path.join(path_data, 'SubjectA_1213_S3_t3v2', 'SubjectA_1213_S3_train')
            elif partition == "valid_3":
                dir_data = os.path.join(path_data, 'SubjectA_1213_S3_t3v2', 'SubjectA_1213_S3_test')
            elif partition == "train_4":
                dir_data = os.path.join(path_data, 'SubjectA_1220_S4_t3v2', 'SubjectA_1220_S4_train')
            elif partition == "valid_4":
                dir_data = os.path.join(path_data, 'SubjectA_1220_S4_t3v2', 'SubjectA_1220_S4_test')


            print(f"Loading data from: {dir_data}")
        
            # ============= Single Run ER Manually - Dataset B =============
            if partition == 'train_ER':
                X_buf = None
                y_buf = None
                seen_examples = 0
                # Dataset used to renew the buffer - this session training data! - Notice that the updated buffer will only be used in the next TL phase.
                buf_dir_data = os.path.join(path_data, 'SubjectA_1220_S4_t3v2', 'SubjectA_1220_S4_train')
                print(f"Loading buffer data from: {buf_dir_data}")
                buffer_size = 200
                if os.path.isfile('buffer_saved/saved_buffer_exp0244_3.pickle'):
                    with open('buffer_saved/saved_buffer_exp0244_3.pickle', 'rb') as handle:
                        saved_buffer = pickle.load(handle)
                    seen_examples = saved_buffer['seen_examples']
                    X_buf = saved_buffer['X_buf']
                    y_buf = saved_buffer['y_buf']
                dataset = BCI_Drone_Dataset_With_Buffer(root=dir_data, bufroot=buf_dir_data, fs=fs, transform=transform, 
                filter_config=filter_config, ds=ds, X_buf=X_buf, y_buf=y_buf, buffer_size=buffer_size, seen_examples=seen_examples, phase=4)
            else:
                dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config=filter_config, ds=ds)
    
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])
    print(dataset[0][0].shape)
    
    return dataset
    

'''
# For calculating max/min/mean/std for dataset: 
if __name__ == "__main__":

    filter_config = {
			"filter_type": "bandpass",
			"fc_low": 0.5,
			"fc_high": 100.0,
			"order": 4
		    }
    
    class TimeWindowPostCue(object):

        def __init__(self, fs, t1_factor, t2_factor):
            self.t1 = int(t1_factor * fs)
            self.t2 = int(t2_factor * fs)

        def __call__(self, sample):
            return sample[:, :, self.t1:self.t2]


    class ReshapeTensor(object):
        def __call__(self, sample):
            return sample.view(1, sample.shape[0], sample.shape[1])
    
    
    transform = Compose([ReshapeTensor(),
	                TimeWindowPostCue(500, 5.1, 8.9)])

    dataset = BCI_Drone_Dataset(root='/path/to/dataset', fs=500, transform=transform, filter_config = filter_config, ds = 1)
    
    full_tensor = []
    for i in range(len(dataset)):
        full_tensor.append(dataset[i][0])
    
    full_tensor = np.array(full_tensor)
    #print(full_tensor.shape)
    
    print("Max: {}".format(full_tensor.max()))
    print("Min: {}".format(full_tensor.min()))
    print("Mean: {}".format(full_tensor.mean()))
    print("Std: {}".format(full_tensor.std()))
    print("eps - 256, symmetric: {}".format(2*max(abs(full_tensor.max()), abs(full_tensor.min()))/256))
'''




