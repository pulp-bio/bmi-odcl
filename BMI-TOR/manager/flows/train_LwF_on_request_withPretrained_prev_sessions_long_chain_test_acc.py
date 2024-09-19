# 
# train_LwF_on_request_withPretrained_prev_sessions_long_chain_test_acc.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

import argparse
import torch
import copy

from manager.platform   import PlatformManager
from manager.logbook    import Logbook
from manager.assistants import DataAssistant
from manager.assistants import NetworkAssistant
from manager.assistants import TrainingAssistant
from manager.assistants import MeterAssistant

from quantlib.editing.fx.passes.pact import PACT_symbolic_trace_inclusive

from manager.flows.pytorchtools import EarlyStopping

import numpy as np
import copy

def train_LwF_on_request_withPretrained_prev_sessions_long_chain_test_acc(args: argparse.Namespace):
    """Train a DNN or (possibly) a QNN in TOR framework with LwF.

    This function implements QuantLab's training flow. The training
    flow applies the mini-batch stochastic gradient descent algorithm,
    or a variant of its, to optimise a target (possibly quantized)
    deep neural network. To corroborate the statistical reliability of
    experimental results, this flow supports :math:`K`-fold
    cross-validation (CV).

    At a high level, this function is structured in two main blocks:
    the *flow preamble* and the *training loop*.

    1. **Flow preamble**.
       The purpose of this part is setting up the bookkeeping
       infrastructure and preparing the software abstractions
       required to train the target DNN system. More in detail, it
       does the following:

       * import the software abstractions (classes, functions)
         required to assemble the learning system from the
         :ref:`systems <systems-package>` package;
       * load the parameters required to instantiate the software
         components of the learning system from the experimental
         unit's private logs folder, that is stored on disk;
       * parse and pass these two pieces of information to
         *assistants*, builder objects that are in charge of
         instantiating the software components of the learning
         system on-demand.

    2. **Training loop**.
       The purpose of this part is performing :math:`K` independent
       *training runs*, one for each fold specified by the CV setup.
       The runs are executed and numbered sequentially starting from
       zero (i.e., QuantLab does not support parallel execution of
       CV folds). Each training run consists of:

       * a *fold preamble*, during which the fold-specific logging
         structure is set up and the software components of the
         learning system are created by the builder objects; these
         components are re-instantiated from scratch (but according
         to the same specified configuration) at the beginning of
         each fold;
       * a *loop over epochs*; during each epoch, the flow will
         perform a loop over batches of training data points while
         optimising (i.e., training) the learning system, then
         perform a loop over batches of validation data points (no
         optimisation is performed during this phase), and, if
         necessary, store a checkpoint of the system's state; during
         the loop, statistics are collected.

    This function implements a checkpointing system to recover from
    unexpected interruptions of the training flow. In case of crashed
    or interrupted training runs, the flow will attempt to resume the
    loop from the most recent checkpoint of the run that was being
    performed when the flow was interrupted. This recovery is
    performed in two steps:

    * at the end of the *preamble*, the flow inspects the logs folder
      looking for the fold logs folder having the largest ID (recall
      that this flow carries out different CV folds sequentially); the
      experiment is resumed from the corresponding training run;
    * during the *fold preamble* of the resumed training run, after
      creating the software components of the learning system, the
      flow inspects the fold's checkpoint folder looking for the most
      recently created checkpoint file; the state of all the software
      components is restored, and the loop can continue from the
      same state it was in before the interruption.
    """

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    platform.show()

    # === FLOW: START ===

    # === FLOW: PREAMBLE ===

    # import the libraries required to assemble the ML system
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

    traindataassistant = DataAssistant('full_sess')
    traindataassistant.recv_datamessage(logbook.send_datamessage('train'))
    validdataassistant_cur = DataAssistant('full_sess_val')
    validdataassistant_cur.recv_datamessage(logbook.send_datamessage('valid'))
    validdataassistant = DataAssistant('valid')
    validdataassistant.recv_datamessage(logbook.send_datamessage('valid'))

    # network
    networkassistant = NetworkAssistant()
    networkassistant.recv_networkmessage(logbook.send_networkmessage())
    # training
    trainingassistant = TrainingAssistant()
    trainingassistant.recv_trainingmessage(logbook.send_trainingmessage())
    # meters
    trainmeterassistant = MeterAssistant('train')
    trainmeterassistant.recv_metermessage(logbook.send_metermessage('train'))
    validmeterassistant = MeterAssistant('valid')
    validmeterassistant.recv_metermessage(logbook.send_metermessage('valid'))
    validmeterassistant_before = MeterAssistant('valid')
    validmeterassistant_before.recv_metermessage(logbook.send_metermessage('valid'))

    # determine the status of cross-validation
    # [recovery] master-workers synchronisation point: find the fold ID by inspecting the experimental unit's logs folder
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.logs_manager.discover_fold_id()
        start_fold_id = logbook.logs_manager.fold_id
    if platform.is_horovod_run:
        start_fold_id = platform.hvd.broadcast_object(start_fold_id, root_rank=platform.master_rank, name='start_fold_id')

    # === LOOP ON FOLDS: START ===

    if args.early_stopping:
        avg_early_stopping_epochs = 0

    # single cycle over CV folds (main loop of the experimental run)
    TL_train = []

    num_train_acc_low = 0
    num_train_acc_high = 0
    train_low_acc_list = []

    test_acc_list_each_fold = [[] for i in range(5)]
    TL_id_test_acc_list_each_fold = [[] for i in range(5)]

    for fold_id in range(start_fold_id, logbook.n_folds):
        group_size = 10
        loader_list = traindataassistant.prepare_on_request_long_chain_multi_sess(platform, fold_id, group_size)
        valid_loader_list = validdataassistant_cur.prepare_on_request(platform, fold_id, group_size)
        prev_valid_loader = validdataassistant.prepare(platform, fold_id)
        num_TL = len(loader_list) - 1
        print("num_TL: {}".format(num_TL))
        cur_TL_train = []

        prev_train_flag = 0
        prev_train_acc = 0

        TL_id = 0

        T_thold = 90

        while TL_id <= num_TL:

            cur_trial_no = TL_id % 10

            print("TL_id: {}".format(TL_id))

            prev_epochs = -2

            # === FOLD: START ===
            # === FOLD: PREAMBLE ===

            # master-only point: prepare fold logs folders
            if (not platform.is_horovod_run) or platform.is_master:
                logbook.logs_manager.setup_fold_logs(fold_id=fold_id)

            # prepare the system components for the current fold
            train_loader = loader_list[TL_id]
            if TL_id != num_TL:
                train_loader_real = loader_list[TL_id+1]

            net = networkassistant.prepare(platform, fold_id)
            # Note: for this step we should have a pretrained model! This pretrained model "net_old" remains unchanged during the training of the full sess.
            net_old = copy.deepcopy(net)
            for param in net_old.parameters():
                param.requires_grad = False

            # training
            loss_fn      = trainingassistant.prepare_loss(net)
            gd           = trainingassistant.prepare_gd(platform, net)
            qnt_ctrls    = trainingassistant.prepare_qnt_ctrls(net)
            # meters
            train_meter  = trainmeterassistant.prepare(platform, len(train_loader), net, gd.opt)
            valid_meter  = validmeterassistant.prepare(platform, len(train_loader), net)
            before_train_valid_meter = validmeterassistant_before.prepare(platform, len(train_loader), net)

            # [recovery] master-workers synchronisation point: load the latest checkpoint from disk
            if (not platform.is_horovod_run) or platform.is_master:
                if TL_id == 0:
                    start_epoch_id = logbook.logs_manager.load_checkpoint(platform, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter=train_meter, valid_meter=valid_meter, ckpt_id=prev_epochs, freeze_except_last=args.freeze_except_last)
                else:
                    start_epoch_id = logbook.logs_manager.load_checkpoint(platform, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter=train_meter, valid_meter=valid_meter, ckpt_id=prev_epochs, freeze_except_last=args.freeze_except_last)
            if platform.is_horovod_run:
                start_epoch_id = platform.hvd.broadcast_object(start_epoch_id, root_rank=platform.master_rank, name='start_epoch_id')
                platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)
                platform.hvd.broadcast_optimizer_state(gd.opt, root_rank=platform.master_rank)
                if gd.lr_sched is not None:
                    sd_lr_sched = platform.hvd.broadcast_object(gd.lr_sched.state_dict(), root_rank=platform.master_rank, name='sd_lr_sched')
                    if not platform.is_master:
                        gd.lr_sched.load_state_dict(sd_lr_sched)
                for i, c in enumerate(qnt_ctrls):
                    sd_c = platform.hvd.broadcast_object(c.state_dict(), root_rank=platform.master_rank, name='sd_c_{}'.format(i))
                    if not platform.is_master:
                        c.load_state_dict(sd_c)
                train_meter.best_loss = platform.hvd.broadcast_object(train_meter.best_loss, root_rank=platform.master_rank, name='train_meter_best_loss')
                valid_meter.best_loss = platform.hvd.broadcast_object(valid_meter.best_loss, root_rank=platform.master_rank, name='valid_meter_best_loss')

            # if no checkpoint has been found, the epoch ID is set to -1
            # [recovery] if a checkpoint has been found, its epoch ID marks a completed epoch; the training should resume from the following epoch
            start_epoch_id += 1

            # === LOOP ON EPOCHS: START ===

            # master-only point: writer stubs are resolved to real TensorBoard writers
            # [recovery] the collected statistics about epochs and iterations carried out after the last stored checkpoint are erased
            if (not platform.is_horovod_run) or platform.is_master:
                logbook.logs_manager.create_writers(start_epoch_id=start_epoch_id, n_batches_train=len(train_loader), n_batches_valid=len(train_loader)) #valid_loader
            
            print()

            try:
                print("[QuantLab] === PyTorch Network ===")
                gm = PACT_symbolic_trace_inclusive(net)
                print(gm.modules)
                gm.graph.print_tabular()
            except:
                print("[QuantLab] === PyTorch Network (non-tracable) ===\n", net)
            
            print()

            if args.freeze_except_last:
                for name, param in net.named_parameters():
                    if param.requires_grad and 'fc' in name:
                        param.requires_grad = True
                    elif param.requires_grad:
                        param.requires_grad = False

            if args.early_stopping:
                early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            if TL_id == 0:
                logbook.logs_manager.store_checkpoint(-1, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter)

            # cycle over epochs (one loop for each fold)
            if TL_id >= 0:
                cur_epochs = start_epoch_id + logbook.config['training']['n_epochs_TL'] # commented if not loading the best.


            net.eval()
            # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
            if (not platform.is_horovod_run) or platform.is_master:
                for c in qnt_ctrls:
                    c.step_pre_training_epoch(start_epoch_id)
                    c.step_pre_validation_epoch(start_epoch_id)
            if platform.is_horovod_run:
                platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

            with torch.no_grad():
                for batch_id, (x, ygt) in enumerate(train_loader):

                    before_train_valid_meter.step(start_epoch_id, batch_id)
                    before_train_valid_meter.start_observing()
                    before_train_valid_meter.tic()

                    # processing (forward pass)
                    x = x.to(platform.device)
                    ypr = net(x)
                    ypr_old = net_old(x)

                    # loss evaluation
                    ygt = ygt.to(platform.device)
                    loss = loss_fn(ypr, ygt, ypr_old)

                    before_train_valid_meter.update(ygt, ypr, loss)
                    before_train_valid_meter.toc(ygt)

            cur_test_acc = before_train_valid_meter.cur_task_statistic
            test_acc_list_each_fold[fold_id].append(cur_test_acc)
            TL_id_test_acc_list_each_fold[fold_id].append(TL_id)
                

            if cur_test_acc < T_thold and cur_trial_no != 9 and TL_id != num_TL:
                print("Start Training! ")

                cur_TL_train.append(TL_id+1)

                net.train()

                for epoch_id in range(start_epoch_id, cur_epochs):

                    # === EPOCH: START ===

                    # === TRAINING STEP: START ===
                    net.train()

                    # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
                    if (not platform.is_horovod_run) or platform.is_master:
                        for c in qnt_ctrls:
                            c.step_pre_training_epoch(epoch_id)
                    if platform.is_horovod_run:
                        platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

                    # cycle over batches of training data (one loop for each epoch)
                    for batch_id, (x, ygt) in enumerate(train_loader_real):
                        # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
                        # TODO: in multi-process runs, synchronising processes at each step might be too costly
                        if (not platform.is_horovod_run) or platform.is_master:
                            for c in qnt_ctrls:
                                c.step_pre_training_batch()
                        if platform.is_horovod_run:
                            platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

                        # event: forward pass is beginning
                        train_meter.step(epoch_id, batch_id)
                        train_meter.start_observing()
                        train_meter.tic()

                        # processing (forward pass)
                        x   = x.to(platform.device)
                        ypr = net(x)

                        ypr_old = net_old(x)

                        # loss evaluation
                        ygt  = ygt.to(platform.device)
                        loss = loss_fn(ypr, ygt, ypr_old)

                        # event: forward pass has ended; backward pass is beginning
                        train_meter.update(ygt, ypr, loss)

                        # training (backward pass)
                        gd.opt.zero_grad()  # clear gradients
                        loss.backward()     # gradient computation
                        gd.opt.step()       # gradient descent

                        # event: backward pass has ended
                        train_meter.toc(ygt)
                        train_meter.stop_observing()

                    # === TRAINING STEP: END ===
                    train_meter.check_improvement()

                    net.eval()

                    # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
                    if (not platform.is_horovod_run) or platform.is_master:
                        for c in qnt_ctrls:
                            c.step_pre_validation_epoch(epoch_id)
                    if platform.is_horovod_run:
                        platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

                    # === EPOCH EPILOGUE ===

                    # (possibly) change learning rate
                    if gd.lr_sched is not None:
                        if isinstance(gd.lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            gd.lr_sched.step(loss)
                        else:
                            gd.lr_sched.step()

                    # has the target metric improved during the current epoch?
                    if logbook.target_loss == 'train':
                        is_best = train_meter.is_best
                    elif logbook.target_loss == 'valid':
                        is_best = valid_meter.is_best

                    # master-only point: store checkpoint to disk if this is a checkpoint epoch or the target metric has improved during the current epoch
                    if (not platform.is_horovod_run) or platform.is_master:
                        if epoch_id % logbook.config['experiment']['ckpt_period'] == 0:  # checkpoint epoch; note that the first epoch is always a checkpoint epoch
                            logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter)
                        if epoch_id == cur_epochs - 1: # logbook.n_epochs - 1:
                            logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter)  # this is the last epoch
                        if is_best:  # the target metric has improved during this epoch
                            logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter, is_best=is_best)

                    if args.early_stopping:
                        early_stopping(valid_meter.cur_loss_statistic, net)
                        if early_stopping.early_stop:
                            print("Early stopping!")
                            avg_early_stopping_epochs += epoch_id + 1
                            logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter, is_early_stop=True)
                            break
                        elif epoch_id == logbook.n_epochs_TL - 1:
                            avg_early_stopping_epochs += epoch_id + 1


                prev_train_flag = 1
                prev_train_acc = train_meter.cur_task_statistic

                if prev_train_acc < T_thold:
                    num_train_acc_low += 1
                    train_low_acc_list.append(prev_train_acc)
                else:
                    num_train_acc_high += 1

                net.eval()
                
                print("Cur epoch: {}".format(epoch_id))


                TL_id = TL_id + 2
                
                # === EPOCH: END ===

            else:
                print("Not Training! ")

                net.eval()

                prev_train_flag = 0

                # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
                if (not platform.is_horovod_run) or platform.is_master:
                    for c in qnt_ctrls:
                        c.step_pre_validation_epoch(start_epoch_id)
                if platform.is_horovod_run:
                    platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

                TL_id = TL_id + 1
                

            # master-only point: when a ``SummaryWriter`` is built it is bound to a fold directory, so when a fold is completed it's time to destroy it
            if (not platform.is_horovod_run) or platform.is_master:
                logbook.logs_manager.destroy_writers()

            # === LOOP ON EPOCHS: END ===

            # reset starting epoch
            start_epoch_id = 0

        TL_train.append(cur_TL_train)

        # === FOLD: END ===

    print("TL_train list: ")
    print(TL_train)

    print("Num num_train_acc_low: {}".format(num_train_acc_low))
    print("Num num_train_acc_high: {}".format(num_train_acc_high))

    print("train_low_acc_list: {}".format(train_low_acc_list))

    print("Test Acc:")
    print(test_acc_list_each_fold)
    print("Test Trial No. List")
    print(TL_id_test_acc_list_each_fold)

    n_sess = 6

    total_acc_per_fold_for_each_sess = [[] for i in range(n_sess)] 
    total_train_trials_per_fold_for_each_sess = [[] for i in range(n_sess)] 

    for i in range(5):
        cur_fold_test_trials = [0 for j in range(n_sess)]
        for ind in range(len(TL_id_test_acc_list_each_fold[i])):
            cur_acc = test_acc_list_each_fold[i][ind]
            cur_TL_id = TL_id_test_acc_list_each_fold[i][ind]
            cur_sess_no = cur_TL_id // 10
            total_acc_per_fold_for_each_sess[cur_sess_no].append(cur_acc)
            cur_fold_test_trials[cur_sess_no] += 1
        for j in range(n_sess):
            total_train_trials_per_fold_for_each_sess[j].append((10-cur_fold_test_trials[j])*10)


    
    avg_acc_per_fold_for_each_sess = [0 for i in range(n_sess)]
    avg_train_trials_per_fold_for_each_sess = [0 for i in range(n_sess)]
    std_acc_per_fold_for_each_sess = [0 for i in range(n_sess)]
    std_train_trials_per_fold_for_each_sess = [0 for i in range(n_sess)]

    for i in range(n_sess):
        avg_acc_per_fold_for_each_sess[i] = np.average(np.array(total_acc_per_fold_for_each_sess[i])) 
        avg_train_trials_per_fold_for_each_sess[i] = np.average(np.array(total_train_trials_per_fold_for_each_sess[i])) 
        std_acc_per_fold_for_each_sess[i] = np.std(np.array(total_acc_per_fold_for_each_sess[i]))
        std_train_trials_per_fold_for_each_sess[i] = np.std(np.array(total_train_trials_per_fold_for_each_sess[i]))

    print("Avg test acc per fold for each sess:")
    print(avg_acc_per_fold_for_each_sess)
    print("Avg num of training trials per fold for each sess:")
    print(avg_train_trials_per_fold_for_each_sess)
    print("Std test acc per fold for each sess:")
    print(std_acc_per_fold_for_each_sess)
    print("Std num of training trials per fold for each sess:")
    print(std_train_trials_per_fold_for_each_sess)

    if args.early_stopping:
        avg_early_stopping_epochs /= logbook.n_folds
        print(f"Average early stopping #epoch: {avg_early_stopping_epochs}")

    # === LOOP ON FOLDS: END ===

    # === FLOW: END ===
