# 
# visionstatistic.py
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

import torch

from systems.BCI_Drone_4classes.utils.transforms.transforms import BCIDRONESTATS
from manager.meter import GrayscaleInputsSnapshot

from manager.platform import PlatformManager
from manager.meter import WriterStub


class BCIDroneInputsSnapshot(GrayscaleInputsSnapshot):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, n_inputs: int, preprocessing_type: int, writer_kwargs: dict = {}):

        super(BCIDroneInputsSnapshot, self).__init__(platform=platform, writerstub=writerstub,
                                                    n_epochs=n_epochs, n_batches=n_batches,
                                                    start=start, period=period,
                                                    name=name, module=module, n_inputs=n_inputs, preprocessing_type=preprocessing_type, preprocessing_stats=BCIDRONESTATS['normalize'], writer_kwargs=writer_kwargs)
