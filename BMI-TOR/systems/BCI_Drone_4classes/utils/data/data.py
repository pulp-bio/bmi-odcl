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
                        
                #recent
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
    count_0 = 0
    count_1 = 0
    for trial_idx in range(number_of_runs * number_of_trials):
        if np.any(X[trial_idx]):
            # Filter out artifacts, but currently no threshold are set.
            max_val = np.max(np.abs(X[trial_idx, :, 2550:4450]))
            X_new_shape.append(X[trial_idx])
            y_new_shape.append(y[trial_idx])
            if y[trial_idx] == 0:
                count_0 += 1
            else:
                count_1 += 1


    X = np.array(X_new_shape)
    y = np.array(y_new_shape)

    print(X.shape)

    print(count_0)
    print(count_1)

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
                  
    if partition in {'train', 'valid', "full_sess", "full_sess_val", "test"}:
        if n_folds > 1 and not use_pseudo_folds:
            # For 5-fold TOR
            if partition == 'valid':  # ER: valid on prev sess
                dir_data = os.path.join(path_data, 'DatasetA_full_data_exc_first')  # ER test: excluding session 1
                print(f"Loading data from: {dir_data}")
                dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config=filter_config, ds=ds)

            elif partition == "full_sess": # ER: train on all sessions (selected indices)
                dir_data = os.path.join(path_data, 'DatasetA_full_data') # cv_data_A_1220_mm 'train_4classes', train_TL_2classes, train_TL_A_1220_MM
                print(f"Loading data from: {dir_data}")
                dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config=filter_config, ds=ds)

            elif partition == "full_sess_val": # ER: all except session 1
                dir_data = os.path.join(path_data, 'DatasetA_full_data_exc_first') # cv_data_A_1220_mm'train_4classes', train_TL_2classes, train_TL_A_1220_MM
                print(f"Loading data from: {dir_data}")
                dataset = BCI_Drone_Dataset(root=dir_data, fs=fs, transform=transform, filter_config=filter_config, ds=ds)

        else:
            # For the pretrained model
            if partition == "train":
                dir_data = os.path.join(path_data, 'DatasetA_S1_1127') # 2022Lan_S1_1127 'train_4classes', train_TL_2classes, train_TL_A_1220_MM
            elif partition == 'valid':
                dir_data = os.path.join(path_data, 'DatasetA_S1_1127') # os.path.join(path_data, 'cv_data_A_MM_3t2v', 'test_TL_A_1220_MM')
            elif partition == 'test':
                dir_data = os.path.join(path_data, 'DatasetA_S1_1127') # os.path.join(path_data, 'cv_data_A_MM_3t2v', 'test_TL_A_1220_MM')
        
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



