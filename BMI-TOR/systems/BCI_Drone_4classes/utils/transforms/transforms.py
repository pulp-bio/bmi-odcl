# 
# transforms.py
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

""" Transform EEG Data before DataLoader. """

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch as t
import torch.nn.functional as F
import random
from torchvision.transforms import Compose

from torchvision.transforms import Lambda

from quantlib.algorithms.pact import PACTAsymmetricAct
from quantlib.algorithms.pact.util import almost_symm_quant

BCIDRONESTATS =\
    {
        'normalize':
            {
                'mean': (-0.001237497664988041, ), # -0.0065582129172980785 # 2.5726935863494873  # 4 A MM sessions: -0.0009483903413638473
                'std':  (4.544654369354248, ) #4.488907814025879 # 20.724111557006836 # 4 A MM sessions: 4.779067039489746
            },
            
        'quantize':
            {
                'min': -81.19220733642578, # A MM full, after removing artefact trials -119.97660064697266 # A MM S4: -573.7626953125, # A MM S3: -30.259586334228516, # A MM S2: -110.9419174194336, # A MM S1: -81.19220733642578, # -81.19220733642578, # -587.19287109375 # 4 A MM sessions: -573.7626953125
                'max': 93.86177062988281, # A MM full, after removing artefact trials 118.04884338378906 # A MM S4: 221.53074645996094, # A MM S3: 44.84843826293945, # A MM S2: 118.0488433837890, # A MM S1: 93.86177062988281, # 93.86177062988281, # 569.14697265625 # 4 A MM sessions: 221.53074645996094
                'eps': 0.7332950830459595  # A MM full, after removing artefact trials 0.9373171925544739 # A MM S4: 4.482521057128906, # A MM S3: 0.3503784239292145 # A MM S2: 0.922256588935852, # A MM S1: 0.7332950830459595 # 0.7332950830459595 # 4.587444305419922 # 4 A MM sessions: 4.482521057128906
            }
    }


    # A MM S4: -573.7626953125, # A MM S3: -30.259586334228516, # A MM S2: -110.9419174194336, # A MM S1: -81.19220733642578
    # A MM S4: 221.53074645996094, # A MM S3: 44.84843826293945, # A MM S2: 118.0488433837890, # A MM S1: 93.86177062988281
    # A MM S4: 4.482521057128906, # A MM S3: 0.3503784239292145 # A MM S2: 0.922256588935852, # A MM S1: 0.7332950830459595

    # A MI S4: MIN: -5232.4638671875
    # A MI S4: MAX: 2548.303955078125
    # A MI S4: EPS: 40.878623962402344

    # A MI S3: MIN: -74.05106353759766
    # A MI S3: MAX: 103.940185546875
    # A MI S3: EPS: 0.8120326995849609

    # A MI S2: MIN: -49.735382080078125
    # A MI S2: MAX: 84.62138366699219
    # A MI S2: EPS: 0.6611045598983765

    # A MI S1: MIN: -53.78331756591797
    # A MI S1: MAX: 50.8947868347168
    # A MI S1: EPS: 0.42018216848373413


BCIDRONESTATS_MULTI =\
    {
        'normalize':
            {
                'mean': (-0.0009483903413638473, ), # -0.0065582129172980785 # 2.5726935863494873  # 4 A MM sessions: -0.0009483903413638473
                'std':  (4.779067039489746, ) #4.488907814025879 # 20.724111557006836 # 4 A MM sessions: 4.779067039489746
            },
            
        'quantize':
            {
                'min': -573.7626953125, # -81.19220733642578, # -587.19287109375 # 4 A MM sessions: -573.7626953125
                'max': 221.53074645996094, # 93.86177062988281, # 569.14697265625 # 4 A MM sessions: 221.53074645996094
                'eps': 4.482521057128906 # 0.7332950830459595 # 4.587444305419922 # 4 A MM sessions: 4.482521057128906
            }
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

class PaddingTensor(object):
    def __init__(self, padding):
        self.padding = int(padding)

    def __call__(self, sample):
        if self.padding == 0:
            return sample

        sample_pad = np.zeros((sample.shape[0], sample.shape[1], sample.shape[2] + self.padding), dtype=np.float32)
        for i in range(sample_pad.shape[0]):
            for ch in range(sample_pad.shape[1]):
                for j in range(sample.shape[2]):
                    sample_pad[i, ch, j] = sample[i, ch, j]

        return sample_pad


class TransformOriginal(Compose):

    def __init__(self, fs : int = 500, t1_factor : float = 5.1, t2_factor : float = 8.9, padding : int = 0):

        if padding is None:
            padding = 0

        transforms = []
        transforms.append(ReshapeTensor())
        transforms.append(TimeWindowPostCue(fs, t1_factor, t2_factor))
        transforms.append(PaddingTensor(padding))
        
        super(TransformOriginal, self).__init__(transforms)
        
        #Compose([ReshapeTensor(),
	#                TimeWindowPostCue(fs, t1_factor, t2_factor),
	#                PaddingTensor(padding)])

class BCIDRONEPACTQuantTransform(Compose):

    """Extend a BCI_Drone (originally: CIFAR10) transform to quantize its outputs.

    The input can be fake-quantized (`quantize == 'fake'`) or true-quantized
    (`quantize == 'int'`).
    """
    def __init__(self, fs : int = 500, t1_factor : float = 5.1, t2_factor : float = 8.9, padding : int = 0, quantize='fake', n_q=256, pad_channels : Optional[int] = None, clip : bool = False):

        transforms = []
        #transforms.append(Transform(augment, crop_size=crop_size, padding=padding))
        transforms.append(ReshapeTensor())
        transforms.append(TimeWindowPostCue(fs, t1_factor, t2_factor))
        transforms.append(PaddingTensor(padding))
        
        if quantize in ['fake', 'int']:
            transforms.append(PACTAsymmetricAct(n_levels=n_q, symm=True, learn_clip=False, init_clip='max', act_kind='identity'))
            quantizer = transforms[-1]
            # set clip_lo to negative max abs of BCI_Drone
            maximum_abs = torch.max(torch.tensor([v for v in BCIDRONESTATS['quantize'].values()]).abs())
            clip_lo, clip_hi = almost_symm_quant(maximum_abs, n_q)
            print("clip_lo and clip_hi: ")
            print(clip_lo)
            print(clip_hi)
            quantizer.clip_lo.data = clip_lo
            quantizer.clip_hi.data = clip_hi
            quantizer.started |= True

        if quantize == 'int':
            eps = transforms[-1].get_eps()
            div_by_eps = lambda x: x/eps
            transforms.append(Lambda(div_by_eps))
        if pad_channels is not None and pad_channels != 3:
            assert pad_channels > 1, "Can't pad MNIST data to <1 channels!"
            pad_img = lambda x: nn.functional.pad(x, (0,0,0,0,0,pad_channels-1), mode='constant', value=0.)
            transforms.append(Lambda(pad_img))
        if clip:
            do_clip = lambda x: nn.functional.relu(x)
            transforms.append(Lambda(do_clip))

        super(BCIDRONEPACTQuantTransform, self).__init__(transforms)
