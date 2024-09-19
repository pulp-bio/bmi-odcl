# 
# test.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# Lan Mei <lanmei@student.ethz.ch>
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

import torch

from manager.platform   import PlatformManager
from manager.logbook    import Logbook
from manager.assistants import DataAssistant
from manager.assistants import NetworkAssistant
from manager.assistants import TrainingAssistant
from manager.assistants import MeterAssistant

from quantlib.algorithms.pact.pact_controllers import *

import numpy as np

import torch.nn as nn

def test_multi(args):

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    # determine the libraries required to assemble the ML system
    logbook = Logbook(args.problem, args.topology)

    # master-only point: in multi-process runs, each process creates a logbook, but only one is privileged enough to interact with the disk
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.boot_logs_manager(exp_id=args.exp_id)

    # master-workers synchronisation point: load configuration from disk
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.config = logbook.logs_manager.load_config()
    if platform.is_horovod_run:
        logbook.config = platform.hvd.broadcast_object(logbook.config, root_rank=platform.master_rank, name='config')

    # prepare assistants
    # data
    testdataassistant = DataAssistant('valid')  # TODO: be sure that `load_data_set` has a valid test branch
    testdataassistant.recv_datamessage(logbook.send_datamessage('valid'))  # TODO: maybe the user would like to test images one at a time, so we should define a new `data` sub-section for test settings
    # network
    networkassistant = NetworkAssistant()
    networkassistant.recv_networkmessage(logbook.send_networkmessage())
    # training
    trainingassistant = TrainingAssistant()
    trainingassistant.recv_trainingmessage(logbook.send_trainingmessage())
    # meters
    testmeterassistant = MeterAssistant('valid')  # TODO: be sure that `MeterAssistant` can create proper test meter (e.g., no LR statistic, no profiling, ...
    testmeterassistant.recv_metermessage(logbook.send_metermessage('valid'))

    TL_acc_list = []
    ckpt_list = args.ckpt_id

    for fold_id in range(0, args.fold_num):

        # master-workers synchronisation point: set fold ID to the one containing the required checkpoint
        if (not platform.is_horovod_run) or platform.is_master:
            logbook.logs_manager.set_fold_id(fold_id=fold_id)
        if platform.is_horovod_run:
            logbook.fold_id = platform.hvd.broadcast_object(logbook.logs_manager._fold_id, root_rank=platform.master_rank, name='fold_id')
        else:
            logbook.fold_id = logbook.logs_manager._fold_id

        # master-only point: prepare fold logs folders
        if (not platform.is_horovod_run) or platform.is_master:
            logbook.logs_manager.setup_fold_logs(fold_id=fold_id)

        # prepare the entities for the test
        # data
        test_loader = testdataassistant.prepare(platform, logbook.fold_id)
        #prepare_on_request_cur_test_selected
        # network
        net         = networkassistant.prepare(platform, logbook.fold_id)
        # training
        loss_fn     = trainingassistant.prepare_loss(net)
        qnt_ctrls   = trainingassistant.prepare_qnt_ctrls(net)
        # meters
        test_meter  = testmeterassistant.prepare(platform, len(test_loader), net)

        # master-workers synchronisation point: load the desired checkpoint from the fold's logs folder
        if (not platform.is_horovod_run) or platform.is_master:
            logbook.epoch_id = logbook.logs_manager.load_checkpoint(platform, net, opt=None, lr_sched=None, qnt_ctrls=qnt_ctrls, ckpt_id=ckpt_list[fold_id])
        if platform.is_horovod_run:
            logbook.epoch_id = platform.hvd.broadcast_object(logbook.epoch_id, root_rank=platform.master_rank, name='epoch_id')
            platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)
            for i, c in enumerate(qnt_ctrls):
                sd_c = platform.hvd.broadcast_object(c.state_dict(), root_rank=platform.master_rank, name='sd_c_{}'.format(i))
                if not platform.is_master:
                    c.load_state_dict(sd_c)

        # === MAIN TESTING LOOP ===
        net.eval()
        # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
        if (not platform.is_horovod_run) or platform.is_master:
            for c in qnt_ctrls:
                c.step_pre_training_epoch(logbook.epoch_id)
                c.step_pre_validation_epoch(logbook.epoch_id)
        if platform.is_horovod_run:
            platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

        with torch.no_grad():

            for batch_id, (x, ygt) in enumerate(test_loader):

                test_meter.step(logbook.epoch_id, batch_id)
                test_meter.start_observing()
                test_meter.tic()

                # processing (forward pass)
                x = x.to(platform.device)
                ypr = net(x)

                # loss evaluation
                ygt = ygt.to(platform.device)
                if not isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss = loss_fn(inputs=ypr, targets=ygt, is_first=True)
                else:
                    loss = loss_fn(ypr, ygt)

                test_meter.update(ygt, ypr, loss)
                test_meter.toc(ygt)
                test_meter.stop_observing()

        TL_acc_list.append(test_meter.cur_task_statistic)

    TL_acc_list = np.array(TL_acc_list)
    print("Final TL acc shape:")
    print(TL_acc_list.shape)
    TL_acc_mean = np.mean(TL_acc_list)
    TL_acc_std = np.std(TL_acc_list)
    print("Final TL acc:")
    
    print("Test result: {:.2f}±{:.2f}%".format(TL_acc_mean, TL_acc_std))
