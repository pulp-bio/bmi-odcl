# 
# LwFLoss.py
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
import torch.nn as nn
#from torch.autograd import Variable

class LwFLoss(nn.Module):

	def __init__(self, **_loss_fn_kwargs):
		super(LwFLoss, self).__init__()
		#self.old_net = old_net
		#self.net = net
		#print(**_loss_fn_kwargs)
		self.T = _loss_fn_kwargs["T"]

	def forward(self, inputs, targets, inputs_prev=None, is_first=False):
		criterion = nn.CrossEntropyLoss()
		loss_CE = criterion(inputs, targets)
		#print(loss_CE)
		if is_first:
			return loss_CE
		else:
			output_new = torch.log_softmax(inputs/self.T, dim=1)   # compute the log of softmax values
			labels_prev = torch.softmax(inputs_prev/self.T, dim=1)
			loss_LwF = torch.sum(output_new * labels_prev, dim=1, keepdim=False)
			loss_LwF = -torch.mean(loss_LwF, dim=0, keepdim=False)
			#print('loss_LwF: ', loss_LwF)
			#print(loss_LwF)

			return loss_LwF + loss_CE
