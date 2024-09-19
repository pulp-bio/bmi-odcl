# 
# taskstatistic.py
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

from manager.meter import TaskStatistic
import manager

from manager.platform import PlatformManager
from manager.meter import WriterStub
from typing import Union, Callable, List


class BCIDroneStatistic(TaskStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool,
                 postprocess_gt_fun: Callable[[torch.Tensor], torch.Tensor],
                 postprocess_pr_fun: Callable[[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]):  # consider the possibility of deep supervision

        name = "Accuracy{}"
        super(BCIDroneStatistic, self).__init__(platform=platform, writerstub=writerstub,
                                               n_epochs=n_epochs, n_batches=n_batches, name=name,
                                               train=train)

        self._total_tracked = None
        self._total_correct = None
        self._total_positive_correct = None
        self._total_negative_correct = None
        self._total_positive_wrong = None
        self._total_negative_wrong = None
        self._value         = None
        self._value_tp      = None
        self._value_tn      = None
        self._value_fp      = None
        self._value_fn      = None
        self._precision          = None
        self._recall_sensitivity = None
        self._specificity        = None

        self._postprocess_gt_fun = postprocess_gt_fun
        self._postprocess_pr_fun = postprocess_pr_fun

    @property
    def value(self):
        return self._value

    def _reset(self):
        self._total_tracked = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._total_correct = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._total_positive_correct = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device) # tongue 
        self._total_negative_correct = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device) # nothing
        self._total_positive_wrong = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device) # tongue 
        self._total_negative_wrong = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device) # nothing
        self._value         = torch.Tensor([0.0]).to(device=self._platform.device)
        self._value_tp      = torch.Tensor([0.0]).to(device=self._platform.device)
        self._value_tn      = torch.Tensor([0.0]).to(device=self._platform.device)
        self._value_fp      = torch.Tensor([0.0]).to(device=self._platform.device)
        self._value_fn      = torch.Tensor([0.0]).to(device=self._platform.device)
        self._precision          = torch.Tensor([0.0]).to(device=self._platform.device)
        self._recall_sensitivity = torch.Tensor([0.0]).to(device=self._platform.device)
        self._specificity        = torch.Tensor([0.0]).to(device=self._platform.device)

    def _stop_observing(self, *args):
        # master-only point: at the end of the epoch, write the running statistics to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(self._tag.format(""), self._value, global_step=self._epoch_id)
                #self._writer.add_scalar(self._tag.format("_TP"), self._value_tp, global_step=self._epoch_id)
                #self._writer.add_scalar(self._tag.format("_TN"), self._value_tn, global_step=self._epoch_id)
                #self._writer.add_scalar(self._tag.format("_FP"), self._value_fp, global_step=self._epoch_id)
                #self._writer.add_scalar(self._tag.format("_FN"), self._value_fn, global_step=self._epoch_id)
                self._writer.add_scalar(self._tag.format("_Precision"), self._precision, global_step=self._epoch_id)
                self._writer.add_scalar(self._tag.format("_Recall"), self._recall_sensitivity, global_step=self._epoch_id)
                self._writer.add_scalar(self._tag.format("_Spec"), self._specificity, global_step=self._epoch_id)


            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def update(self, ygt: torch.Tensor, ypr: torch.Tensor):

        # adapt the ground truth topology labels and the network predictions to the topology-agnostic statistic
        pp_ygt = self._postprocess_gt_fun(ygt)
        pp_ypr = self._postprocess_pr_fun(ypr)

        # compute the batch statistics for the current process
        bs        = torch.Tensor([ypr.shape[0]]).to(dtype=torch.int64, device=self._platform.device)
        correct   = pp_ygt == pp_ypr
        wrong     = pp_ygt != pp_ypr
        positive_correct = correct & (pp_ypr == 1)
        negative_correct = correct & (pp_ypr == 0)
        positive_wrong = wrong & (pp_ypr == 1)
        negative_wrong = wrong & (pp_ypr == 0)
        n_correct = torch.sum(correct[:, 0])
        n_positive_correct = torch.sum(positive_correct[:, 0])
        n_negative_correct = torch.sum(negative_correct[:, 0])
        n_positive_wrong   = torch.sum(positive_wrong[:, 0])
        n_negative_wrong   = torch.sum(negative_wrong[:, 0])

        # master-workers synchronisation point: different processes apply the model to different data, hence they observe different statistics
        if self._platform.is_multiproc_horovod_run:
            sum_bs        = self._platform.hvd.allreduce(bs,        op=self._platform.hvd.Sum, name='/'.join([self._tag, 'bs']))
            sum_n_correct = self._platform.hvd.allreduce(n_correct, op=self._platform.hvd.Sum, name=self._tag.format(""))
            sum_n_positive_correct = self._platform.hvd.allreduce(n_positive_correct, op=self._platform.hvd.Sum, name=self._tag.format("_TP"))
            sum_n_negative_correct = self._platform.hvd.allreduce(n_negative_correct, op=self._platform.hvd.Sum, name=self._tag.format("_TN"))
            sum_n_positive_wrong   = self._platform.hvd.allreduce(n_positive_wrong, op=self._platform.hvd.Sum, name=self._tag.format("_FP"))
            sum_n_negative_wrong   = self._platform.hvd.allreduce(n_negative_wrong, op=self._platform.hvd.Sum, name=self._tag.format("_FN"))
        else:
            sum_bs        = bs
            sum_n_correct = n_correct
            sum_n_positive_correct = n_positive_correct
            sum_n_negative_correct = n_negative_correct
            sum_n_positive_wrong   = n_positive_wrong
            sum_n_negative_wrong   = n_negative_wrong

        # update running statistics
        self._total_tracked += sum_bs
        self._total_correct += sum_n_correct
        self._total_positive_correct += sum_n_positive_correct
        self._total_negative_correct += sum_n_negative_correct
        self._total_positive_wrong += sum_n_positive_wrong
        self._total_negative_wrong += sum_n_negative_wrong
        self._value          = (100.0 * self._total_correct) / self._total_tracked
        self._value_tp           = (100.0 * self._total_positive_correct) / self._total_tracked
        self._value_tn           = (100.0 * self._total_negative_correct) / self._total_tracked
        self._value_fp           = (100.0 * self._total_positive_wrong) / self._total_tracked
        self._value_fn           = (100.0 * self._total_negative_wrong) / self._total_tracked
        self._precision          = self._value_tp / (self._value_tp + self._value_fp)
        self._recall_sensitivity = self._value_tp / (self._value_tp + self._value_fn)
        self._specificity        = self._value_tn / (self._value_tn + self._value_fp)

        # master-only point: print the running statistic to screen
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            message = manager.QUANTLAB_PREFIX + "Epoch [{:3d}/{:3d}] : Batch [{:5d}/{:5d}] - Accuracy: {:6.2f}%, Precision: {:6.2f}, Recall: {:6.2f}, Specificity: {:6.2f}".format(self._epoch_id, self._n_epochs-1, self._batch_id, self._n_batches, self._value.item(), self._precision.item(), self._recall_sensitivity.item(), self._specificity.item())
            print(message)
