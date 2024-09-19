# 
# quantops.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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
# 
# SPDX-License-Identifier: Apache-2.0
# 

import quantlib.algorithms as qa
import torch
from torch import nn
from torch.nn.parameter import Parameter

class STEActivationInteger(qa.ste.STEActivation):

    def __init__(self, num_levels=2**8-1, frac_bits=17, clamp_min_to_zero=True, is_input_integer=True):
        """Quantization function implementation.

        This object is meant to replace its fake-quantized counterpart which
        is used at training time (defined in the library module
        `quantlib.algorithms.ste.ste_ops`) with a true-quantized version which
        emulates the accelerators' integer arithmetic by embedding it in
        floating-point arithmetic.

        Args:
            num_levels (int): The number of quantization levels
            frac_bits (int): The accelerator's beta-accumulator is a
                fixed-point signed integer. This parameter counts the number
                of fractional bits in the representation
            clamp_min_to_zero (bool): Whether the minimum activation level is
                represented by zero
            is_input_integer (bool): Whether the input to this node comes from
                a quantized or a floating-point layer

        """

        super(STEActivationInteger, self).__init__(num_levels=num_levels, quant_start_epoch=0)

        self.frac_bits = frac_bits
        self.min = 0 if clamp_min_to_zero else (-(num_levels - 1) // 2)  # if STE has been places after ReLU, levels below zero should be clamped
        self.max = (num_levels - 1) // 2
        self.is_input_integer = is_input_integer

    def forward(self, x):

        if self.is_input_integer:
            x = (x / 2**self.frac_bits).floor()

        return x.clamp(min=self.min, max=self.max).floor().detach()


class QuantLayer(nn.Module):
    # signed quantization to n bits
    def __init__(self, scale, n_bits):
        super(QuantLayer, self).__init__()
        self.scale = torch.tensor(scale, dtype=torch.float)
        self.n_bits = n_bits
        self.max_abs = torch.tensor(2**(n_bits-1)-1, dtype=torch.float)
        self.step = Parameter(self.scale/self.max_abs, requires_grad=False)

    def forward(self, x):
        x = torch.clamp(x, -self.scale.item(), self.scale.item())
        unrounded = x/self.step
        return torch.round(unrounded)

class DequantLayer(nn.Module):
    # reverse of QuantLayer
    def __init__(self, scale, n_bits, validate=False):
        super(DequantLayer, self).__init__()
        self.scale = torch.tensor(scale, dtype=torch.float)
        self.n_bits = n_bits
        self.max_abs = torch.tensor(2**(n_bits-1)-1, dtype=torch.float)
        self.step = Parameter(self.scale/self.max_abs, requires_grad=False)
        self.validate = validate

    def forward(self, x):
        # check if x is really an integer
        if self.validate:
            assert torch.sum(x % 1.0 != 0) == 0, "Input to DequantLayer was not integer: {}".format(x)
            assert torch.sum(torch.abs(x) > self.max_abs) == 0, "Some elements of x are larger than {}!".format(self.max_abs)
        # don't clamp this
        return x*self.step

