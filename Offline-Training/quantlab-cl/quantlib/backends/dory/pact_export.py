#
# pact_export.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Lan Mei <lanmei@student.ethz.ch>
#
# Copyright (c) 2020-2024 ETH Zurich.
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

from functools import partial
from itertools import chain 
from pathlib import Path
import json

import random


import torch

from torch import nn, fx
import torchvision

import onnx
import numpy as np

import torch.optim as optim

import quantlib.editing.fx as qlfx
from quantlib.editing.fx.util import module_of_node
from quantlib.editing.lightweight import LightweightGraph
from quantlib.algorithms.pact import RequantShift, PACTIntegerLayerNorm, PACTIntegerGELU, PACTWrapMHSA, PACTWrapModule
from quantlib.algorithms.pact.pact_ops import *
from .dory_passes import AvgPoolWrap, DORYAdder, DORYHarmonizePass


cur_input_idx = -1
start_connected = False
#out_size=2
#in_size=928

def get_input_channels(net : fx.GraphModule):
    for node in net.graph.nodes:
        if node.op == 'call_module' and isinstance(module_of_node(net, node), (nn.Conv1d, nn.Conv2d)):
            conv = module_of_node(net, node)
            return conv.in_channels

# annotate:
#   conv, FC nodes:
#     - 'weight_bits'
#     - 'bias_bits'
#   clip nodes (these represent [re]quantization nodes):
#     - 'out_bits'
#   Mul nodes:
#     - 'mult_bits' - this describes the multiplicative factor as well as the
#                    result's precision
#   Add nodes:
#     - 'add_bits' - this describes the added number's as well as the result's precision



def get_attr_by_name(node, name):
    try:
        a = [attr for attr in node.attribute if attr.name == name][0]
    except IndexError:
        a = "asdf"
    return a


def annotate_onnx(m, prec_dict : dict, requant_bits : int = 32):

# annotate all clip nodes - use the clip limits to determine the number of
# bits; in a properly exported model this would be done based on
# meta-information contained in the pytorch graph.
    clip_nodes = [n for n in m.graph.node if n.op_type == "Clip"]
    for i, n in enumerate(clip_nodes):
        lower = get_attr_by_name(n, "min").f
        #assert lower == 0.0, "clip node {} has lower clip bound {} not equal to zero!".format(n.name, lower)
        upper = get_attr_by_name(n, "max").f
        n_bits = int(np.round(np.log2(upper-lower+1.0)))
        n_bits = n_bits if n_bits <= 8 else 32
        precision_attr = onnx.helper.make_attribute(key='out_bits', value=n_bits)
        n.attribute.append(precision_attr)


    conv_fc_nodes = [n for n in m.graph.node if n.op_type in ['Conv', 'Gemm', 'MatMul']]
    for n in conv_fc_nodes:
        if n.op_type == 'MatMul':
            import ipdb; ipdb.set_trace()
        weight_name = n.input[1].rstrip('.weight')
        weight_bits = prec_dict[weight_name]
        weight_attr = onnx.helper.make_attribute(key='weight_bits', value=weight_bits)
        n.attribute.append(weight_attr)
        # bias accuracy hardcoded to 32b.
        bias_attr = onnx.helper.make_attribute(key='bias_bits', value=32)
        n.attribute.append(bias_attr)

    # assume that add nodes have 32b precision?? not specified in the name...
    add_nodes_requant = [n for n in m.graph.node if n.op_type == 'Add' and not all(i.isnumeric() for i in n.input)]
    # the requantization add nodes are the ones not adding two operation nodes'
    # outputs together
    for n in add_nodes_requant:
        add_attr = onnx.helper.make_attribute(key='add_bits', value=requant_bits)
        n.attribute.append(add_attr)

    add_nodes_residual = [n for n in m.graph.node if n.op_type == 'Add' and n not in add_nodes_requant]
    for n in add_nodes_residual:
        # assume that all residual additions are executed in 8b
        add_attr = onnx.helper.make_attribute(key='add_bits', value=8)
        n.attribute.append(add_attr)

    mult_nodes = [n for n in m.graph.node if n.op_type == 'Mul']
    for n in mult_nodes:
        mult_attr = onnx.helper.make_attribute(key='mult_bits', value=requant_bits)
        n.attribute.append(mult_attr)

def export_net(net : nn.Module,name : str, out_dir : str, eps_in : float, in_data : torch.Tensor, full_in_data : list, full_in_label : list = [], integerize : bool = True,
     D : float = 2**24, opset_version : int  = 10, align_avg_pool : bool = False, code_size : int = 160000,  num_inputs_exported : int = 1, connect_network: bool = False,
     connect_train_input_full : list = [], connect_train_input_full_label : list = [], connect_val_input_full : list = [], connect_val_input_full_label : list = [], 
     connect_test_input_full : list = [], connect_test_input_full_label : list = [], qnet_fcl : nn.Module = None, left_FCL_connect_network : bool = False):

    net = net.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = f"{name}_ql_integerized.onnx"
    net_name = name
    onnx_path = out_path.joinpath(onnx_file)

    shape_in = in_data.shape

    if integerize:
        net_traced = qlfx.passes.pact.PACT_symbolic_trace(net)

        int_pass = qlfx.passes.pact.IntegerizePACTNetPass(shape_in=shape_in,  eps_in=eps_in, D=D)
        net_integerized = int_pass(net_traced)
    else: # assume the net is already integerized
        net_integerized = net

    if align_avg_pool:
        align_avgpool_pass = DORYHarmonizePass(in_shape=shape_in)
        net_integerized = align_avgpool_pass(net_integerized)

    integerized_nodes = LightweightGraph.build_nodes_list(net_integerized, leaf_types=(AvgPoolWrap, DORYAdder, PACTWrapMHSA))

    # the integerization pass annotates the conv layers with the number of
    # weight levels. from this information we can make a dictionary of the number of
    # weight bits.
    prec_dict = {}

    net_integerized.eval()

    for lname, module in net_integerized.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            n_bits = int(np.log2(module.n_levels+1.2))
            prec_dict[lname] = n_bits
            
        #if isinstance(module, (nn.Conv2d)):
            #if lname == "_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0":
                #print(f"-----------{lname} Weights---------")
                #print(module.weight)
                #print(module.weight.shape)
                #print(module.n_levels)
                

    #first export an unannotated ONNX graph
    test_input = torch.rand(shape_in)

    torch.onnx.export(net_integerized.to('cpu'),
                      test_input,
                      str(onnx_path),
                      export_params=True,
                      verbose=True,
                      opset_version=10,
                      do_constant_folding=True,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)


    #load the exported model and annotate it
    onnx_model = onnx.load(str(onnx_path))
    annotate_onnx(onnx_model, prec_dict)
    # finally, save the annotated ONNX model
    onnx.save(onnx_model, str(onnx_path))
    # now we pass a test input through the model and log the intermediate
    # activations

    # make a forward hook to dump outputs of RequantShift layers
    acts = []
    acts_input = []
    output_before_lin = []
    
    def dump_hook(self, inp, outp, name):
        # DORY wants HWC tensors
        global start_connected
        if not start_connected:
            acts_input.append((name, inp[0]))
            acts.append((name, torch.floor(outp[0])))
    # to correctly export unwrapped averagePool nodes, floor all inputs to all nodes

    def floor_hook(self, inp):
        return tuple(torch.floor(i) for i in inp) # tuple(torch.floor(i) for i in inp) # torch.floor(i)
        # return tuple(i.int().double() for i in inp)

    def print_hook(self, inp, outp, name):
        print(torch.floor(outp[0]))
        print(outp[0].shape)

    def before_lin_hook(self, inp, outp, name):
        #print(inp[0])
        #print(inp[0].shape)
        global cur_input_idx
        global start_connected
        if not start_connected:
            #torch.save(inp[0], out_path.joinpath('inputs_fc.pt'))
            np.save(out_path.joinpath(f'inputs_fc_{cur_input_idx}.npy'), inp[0].detach().numpy())
            cur_input_idx += 1
        else:
            output_before_lin.append(inp[0].detach().numpy())

    if left_FCL_connect_network:
        def after_flatten_hook(self, inp, outp, name):
            #print(inp[0])
            #print(inp[0].shape)
            global cur_input_idx
            global start_connected
            if not start_connected:
                #torch.save(inp[0], out_path.joinpath('inputs_fc.pt'))
                np.save(out_path.joinpath(f'inputs_fc_{cur_input_idx}.npy'), inp[0].detach().numpy())
                cur_input_idx += 1
            else:
                output_before_lin.append(outp[0].detach().numpy())

    for n in integerized_nodes:
        #print("Integerized: {}".format(n))
        
        if isinstance(n.module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, AvgPoolWrap, nn.ZeroPad2d, nn.Flatten)): 
            #if n.name == "sep_conv_pad":
            #    continue
            #print("Hooked: {}".format(n))
            print(n.module)
            print(n.name)
            n.module.register_forward_pre_hook(floor_hook)
        #n.module.register_forward_pre_hook(floor_hook)

        if n.name == "before_conv1_pad":
            hook = partial(print_hook, name=n.name)
            n.module.register_forward_hook(hook)

        if isinstance(n.module, (RequantShift, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.Linear, AvgPoolWrap, DORYAdder, PACTIntegerLayerNorm, PACTIntegerGELU, PACTWrapMHSA)): 
            hook = partial(dump_hook, name=n.name)
            n.module.register_forward_hook(hook)

        if isinstance(n.module, (nn.Linear)):
            hook = partial(before_lin_hook, name=n.name)
            n.module.register_forward_hook(hook)

        if left_FCL_connect_network:
            if isinstance(n.module, (nn.Flatten)):
                hook = partial(after_flatten_hook, name=n.name)
                n.module.register_forward_hook(hook)

    # open the supplied input image
    if in_data is not None:
        im_tensor = in_data.clone().to(dtype=torch.float64)
        im_tensor_mod = im_tensor.to(torch.int8).to(dtype=torch.float64)
        net_integerized = net_integerized.to(dtype=torch.float64)
        output = net_integerized(im_tensor_mod).to(dtype=torch.float64)

        for lname, module in net_integerized.named_modules():
            if isinstance(module, (nn.Linear)):
                #print(f"--------------Weights of {lname}-----------")
                #print(module.weight)
                #print(module.weight.shape)
                #print(module.bias)
                torch.save(module.weight, out_path.joinpath('weights_fc.pt'))
                np.save(out_path.joinpath('weights_fc.npy'), module.weight.detach().numpy())

                torch.save(module.bias, out_path.joinpath('bias_fc.pt'))
                np.save(out_path.joinpath('bias_fc.npy'), module.bias.detach().numpy())

        #print("-----------Input---------")
        #print(im_tensor.shape)
        #print(im_tensor)
        #print(im_tensor[0,0,:,0])
        #print("-----------Floored input----------")
        #print(torch.floor(im_tensor[0,0,:,0]))
        #print("-----------Int input----------")
        #for i in im_tensor[0,0,:,0]:
        #    print(int(i))
            
        #print("-----------Uint8 input----------")
        #for i in im_tensor_mod[0,0,:,0]:
        #    print(i)
            
        # now, save everything into beautiful text files
        def save_beautiful_text(t : torch.Tensor, layer_name : str, filename : str):
            t = t.squeeze(0)

            if t.dim()==3:
                # expect a (C, H, W) tensor - DORY expects (H, W, C)
                t = t.permute(1,2,0)
            #SCHEREMO: HACK HACK HACK HACK HACK HACK
#             elif t.dim()==2:
#                 # expect a (C, D) tensor - DORY expects (D, C)
#                 t = t.permute(1,0)
            else:
                print(f"Not permuting output of layer {layer_name}...")

            filepath = out_path.joinpath(f"{filename}.txt")
            with open(str(filepath), 'w') as fp:
                fp.write(f"# {layer_name} (shape {list(t.shape)}),\n")
                for el in t.flatten():
                    fp.write(f"{int(el)},\n") #int()

        '''
        for lname, module in net_integerized.named_modules():
            if isinstance(module, (nn.Conv2d)):
                if lname == "_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2":
                
                    print(module)
                    print(f"--------------Weights of {lname}-----------")
                    print(module.weight)
                    print(module.bias)
                    
                    print(module.weight.shape)
                    '''
        
        save_beautiful_text(im_tensor, "input", "input")
        save_beautiful_text(output, "output", "output")

        for i, (lname, t) in enumerate(acts):
            print("out_layer{}: {}".format(i, lname))
            save_beautiful_text(t, lname, f"out_layer{i}")

    #if num_inputs_exported > 1:  Cannot delete now, as to output intermediate results of DORY network for more than one samples.
    if full_in_data:
        for i in range(num_inputs_exported):
            acts = []
            acts_input = []

            cur_im_tensor = full_in_data[i].clone().to(dtype=torch.float64)
            save_beautiful_text(cur_im_tensor, f"input_{i}", f"input_{i}")
            cur_im_tensor_mod = cur_im_tensor.to(torch.int8).to(dtype=torch.float64)
            net_integerized = net_integerized.to(dtype=torch.float64) # Added!
            cur_output = net_integerized(cur_im_tensor_mod).to(dtype=torch.float64)

            save_beautiful_text(cur_output, f"output_{i}", f"output_{i}")
            for j, (lname, t) in enumerate(acts):
                print("out_layer{}: {}".format(j, lname))
                save_beautiful_text(t, lname, f"out_{i}_layer{j}")
        
    #for i, (lname, t) in enumerate(acts_input):
    #    if (lname == '_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2'):
    #        print("out_layer{}: {}".format(i, lname))
    #        save_beautiful_text(t, lname, f"extra{i}")

    if connect_network or left_FCL_connect_network:
        lr = 0.001
        num_epochs = 100 # 100
        global start_connected
        start_connected = True

        weights_fc = np.array([])
        bias_fc = np.array([])

        out_size = 2
        in_size = 928

        class LinLayer (nn.Module):
            def __init__(self):
                super(LinLayer, self).__init__()
                self.lin = nn.Linear(in_features=in_size, out_features=out_size, bias=True)

            def forward(self, x):
                out = self.lin(x)
                return out

        new_fcl = LinLayer()

        if connect_network:
            for lname, module in net_integerized.named_modules():
                if isinstance(module, (nn.Linear)):
                    weights_fc = module.weight.detach().numpy()
                    bias_fc = module.bias.detach().numpy()
                    out_size = weights_fc.shape[0]
                    in_size = weights_fc.shape[1]

        elif left_FCL_connect_network:
            weights_fc = qnet_fcl.weight.detach().numpy()
            bias_fc = qnet_fcl.bias.detach().numpy()

        print("Integrated mixed-network: ")
        print(net_integerized)
        # Define and initialize new linear layer

        eps_in = 0.0039215689 # From previous session, pretrained model, (obtained from without connect_network)
        eps_w = 0.0007904182 # From previous session, pretrained model
        # TODO: update eps per dataset!

        cur_TLcheckpoint_idx = 2
        load_ckpt = (cur_TLcheckpoint_idx != 0)

        if load_ckpt:
            PATH_load_ckpt = out_path.joinpath(f'new_fcl_TL_{cur_TLcheckpoint_idx}.pth')
            new_fcl.load_state_dict(torch.load(PATH_load_ckpt))
            print(f"Loading model from new_fcl_TL_{cur_TLcheckpoint_idx}.pth... ")
            cur_TLcheckpoint_idx += 1
        else: 
            initial_weights = torch.from_numpy(np.squeeze(weights_fc)).to(torch.float32)
            #print(initial_weights)
            initial_weights = torch.reshape((torch.transpose(torch.reshape(initial_weights, (2,32,29)), 1, 2)), (2,928))
            if connect_network:
                for i in range(out_size):
                    for j in range(in_size):
                        initial_weights[i][j] = initial_weights[i][j] * eps_w

            initial_bias = torch.from_numpy(np.squeeze(bias_fc)).to(torch.float32)
            if connect_network:
                for i in range(out_size):
                    initial_bias[i] = initial_bias[i] * eps_w * eps_in

            new_fcl.lin.weight = nn.Parameter(initial_weights)
            new_fcl.lin.bias = nn.Parameter(initial_bias)
            cur_TLcheckpoint_idx += 1

        print("Cur FCL parameters: ")
        for name, parameter in new_fcl.named_parameters():
            print(name, parameter, parameter.shape)

        new_fcl.zero_grad()
        # Optimizer and criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(new_fcl.parameters(), lr=lr, momentum=0)

        # training process
        num_train_inputs = len(connect_train_input_full)
        num_val_inputs = len(connect_val_input_full)
        num_test_inputs = len(connect_test_input_full)

        # Before TL: Build input lists before FCL - Test
        output_before_lin = []
        for i in range(num_test_inputs):
            cur_im_tensor = connect_test_input_full[i].clone().to(dtype=torch.float64)
            cur_im_tensor_mod = torch.round(cur_im_tensor).to(torch.int8).to(dtype=torch.float64)
            net_integerized = net_integerized.to(dtype=torch.float64)
            cur_output = net_integerized(cur_im_tensor_mod).to(dtype=torch.float64)
        assert len(output_before_lin) == num_test_inputs

        # Before TL: test process on pretrained model, directly after converting from INT8 to FP32, or after loading the checkpoint.
        new_fcl.eval()
        sample_labels = connect_test_input_full_label
        num_correct_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for input_idx in range(num_test_inputs):
                #print("\nCur validation input idx is: ", input_idx)
                indata = torch.from_numpy(np.squeeze(output_before_lin[input_idx])).to(torch.float32) # inputs_fc_val_{input_idx}.npy
                indata = torch.flatten(torch.transpose(torch.reshape(indata, (32,29)), 0, 1)) #torch.flatten(torch.reshape((32,29)).transpose(indata, 0, 1))
                for i in range(in_size):
                    indata[i] = indata[i] * eps_in
                #print("\nInput data is: ", indata, indata.shape, indata.dtype)

                cur_label = sample_labels[input_idx]
                label = torch.zeros(1, out_size)
                label[0][sample_labels[input_idx]] = 1 

                #print("\nValidation Label is: ", cur_label)

                #new_fcl.eval()
                output_predict = new_fcl(indata)
                output = output_predict.unsqueeze(dim=0)

                loss = criterion(output, label)
                test_loss += loss

                predicted_label = np.argmax(output_predict.detach().numpy())
                if cur_label == predicted_label:
                    num_correct_test += 1
                
                #print("\nNet output is: ", output, output.shape, output.dtype)
                #print("\nLoss is: ", loss, loss.shape, loss.dtype)

        cur_test_loss = test_loss / num_test_inputs
        cur_test_acc = num_correct_test / num_test_inputs
        print(f"Before TL (Pretrained model): Test Loss: {cur_test_loss}, Acc: {cur_test_acc}")

        # Build input lists before FCL - Train
        output_before_lin = []
        for i in range(num_train_inputs):
            cur_im_tensor = connect_train_input_full[i].clone().to(dtype=torch.float64)
            cur_im_tensor_mod = torch.round(cur_im_tensor).to(torch.int8).to(dtype=torch.float64)
            net_integerized = net_integerized.to(dtype=torch.float64)
            cur_output = net_integerized(cur_im_tensor_mod).to(dtype=torch.float64)

        print(len(output_before_lin))
        print(num_train_inputs)
        assert len(output_before_lin) == num_train_inputs

        # Build input lists before FCL - Valid
        #output_before_lin = []
        for i in range(num_val_inputs):
            cur_im_tensor = connect_val_input_full[i].clone().to(dtype=torch.float64)
            cur_im_tensor_mod = torch.round(cur_im_tensor).to(torch.int8).to(dtype=torch.float64)
            net_integerized = net_integerized.to(dtype=torch.float64)
            cur_output = net_integerized(cur_im_tensor_mod).to(dtype=torch.float64)
        assert len(output_before_lin) == num_train_inputs + num_val_inputs

        train_samples_output_before_lin = output_before_lin[:num_train_inputs]
        merged_train_sample_label = list(zip(train_samples_output_before_lin, connect_train_input_full_label))
        random.shuffle(merged_train_sample_label)
        train_samples_output_before_lin, connect_train_input_full_label = zip(*merged_train_sample_label)

        for i_epochs in range(num_epochs):
            new_fcl.train()
            sample_labels = connect_train_input_full_label
            num_correct_train = 0
            train_loss = 0.0
            for input_idx in range(num_train_inputs):
                #print("\nCur training input idx is: ", input_idx)
                indata = torch.from_numpy(np.squeeze(train_samples_output_before_lin[input_idx])).to(torch.float32)
                indata = torch.flatten(torch.transpose(torch.reshape(indata, (32,29)), 0, 1))
                for i in range(in_size):
                    indata[i] = indata[i] * eps_in
                indata.requires_grad = True
                #print("\nInput data is: ", indata, indata.shape, indata.dtype)

                cur_label = sample_labels[input_idx]
                label = torch.zeros(1, out_size)
                label[0][sample_labels[input_idx]] = 1 

                #print("\nLabel is: ", label, label.shape, label.dtype)

                # Do a forward computation
                #print("\nCur FCL bias is: ", new_fcl.lin.bias)
                optimizer.zero_grad()
                #output = new_fcl(indata).unsqueeze(dim=0)
                output_predict = new_fcl(indata)
                output = output_predict.unsqueeze(dim=0)

                predicted_label = np.argmax(output_predict.detach().numpy())
                if cur_label == predicted_label:
                    num_correct_train += 1

                # print("\nNet output is: ", output, output.shape, output.dtype)
                # print(output)
                # print(label)
                loss = criterion(output, label)
                # print("\nLoss is: ", loss, loss.shape, loss.dtype)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                #print("\nLoss is: ", loss, loss.shape, loss.dtype)

                '''
                if i_epochs == num_epochs - 1:
                    print("\nFCL output is: ", output, output.shape, output.dtype)
                    print("\nFCL gradients are: ")
                    for name, parameter in new_fcl.named_parameters():
                        print(name, parameter.grad, parameter.grad.shape, parameter.grad.dtype)
                '''

            cur_train_loss = train_loss / num_train_inputs
            cur_train_acc = num_correct_train / num_train_inputs
            print(f"Training Epoch {i_epochs+1}: Loss: {cur_train_loss}, Acc: {cur_train_acc}")

            # validation process
            new_fcl.eval()
            sample_labels = connect_val_input_full_label
            num_correct_val = 0
            val_loss = 0.0

            with torch.no_grad():
                for input_idx in range(num_val_inputs):
                    #print("\nCur validation input idx is: ", input_idx)
                    indata = torch.from_numpy(np.squeeze(output_before_lin[input_idx+num_train_inputs])).to(torch.float32) # inputs_fc_val_{input_idx}.npy
                    indata = torch.flatten(torch.transpose(torch.reshape(indata, (32,29)), 0, 1)) #torch.flatten(torch.reshape((32,29)).transpose(indata, 0, 1))
                    for i in range(in_size):
                        indata[i] = indata[i] * eps_in
                    #print("\nInput data is: ", indata, indata.shape, indata.dtype)

                    cur_label = sample_labels[input_idx]
                    label = torch.zeros(1, out_size)
                    label[0][sample_labels[input_idx]] = 1 

                    #print("\nValidation Label is: ", cur_label)

                    #new_fcl.eval()
                    output_predict = new_fcl(indata)
                    output = output_predict.unsqueeze(dim=0)

                    loss = criterion(output, label)
                    val_loss += loss

                    predicted_label = np.argmax(output_predict.detach().numpy())
                    if cur_label == predicted_label:
                        num_correct_val += 1
                    
                    #print("\nNet output is: ", output, output.shape, output.dtype)
                    #print("\nLoss is: ", loss, loss.shape, loss.dtype)

            cur_val_loss = val_loss / num_val_inputs
            cur_val_acc = num_correct_val / num_val_inputs
            print(f"Validation Epoch {i_epochs+1}: Loss: {cur_val_loss}, Acc: {cur_val_acc}")

        # Build input lists before FCL - Test
        output_before_lin = []
        for i in range(num_test_inputs):
            cur_im_tensor = connect_test_input_full[i].clone().to(dtype=torch.float64)
            cur_im_tensor_mod = torch.round(cur_im_tensor).to(torch.int8).to(dtype=torch.float64)
            net_integerized = net_integerized.to(dtype=torch.float64)
            cur_output = net_integerized(cur_im_tensor_mod).to(dtype=torch.float64)
        assert len(output_before_lin) == num_test_inputs

        # test process
        new_fcl.eval()
        sample_labels = connect_test_input_full_label
        num_correct_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for input_idx in range(num_test_inputs):
                #print("\nCur validation input idx is: ", input_idx)
                indata = torch.from_numpy(np.squeeze(output_before_lin[input_idx])).to(torch.float32) # inputs_fc_val_{input_idx}.npy
                indata = torch.flatten(torch.transpose(torch.reshape(indata, (32,29)), 0, 1)) #torch.flatten(torch.reshape((32,29)).transpose(indata, 0, 1))
                for i in range(in_size):
                    indata[i] = indata[i] * eps_in
                #print("\nInput data is: ", indata, indata.shape, indata.dtype)

                cur_label = sample_labels[input_idx]
                label = torch.zeros(1, out_size)
                label[0][sample_labels[input_idx]] = 1 

                #print("\nValidation Label is: ", cur_label)

                #new_fcl.eval()
                output_predict = new_fcl(indata)
                output = output_predict.unsqueeze(dim=0)

                loss = criterion(output, label)
                test_loss += loss

                predicted_label = np.argmax(output_predict.detach().numpy())
                if cur_label == predicted_label:
                    num_correct_test += 1
                
                #print("\nNet output is: ", output, output.shape, output.dtype)
                #print("\nLoss is: ", loss, loss.shape, loss.dtype)

        cur_test_loss = test_loss / num_test_inputs
        cur_test_acc = num_correct_test / num_test_inputs
        print(f"After TL: Test Loss: {cur_test_loss}, Acc: {cur_test_acc}")

        torch.save(new_fcl.state_dict(), out_path.joinpath(f'new_fcl_TL_{cur_TLcheckpoint_idx}.pth'))

        print("Final FCL parameters: ")
        for name, parameter in new_fcl.named_parameters():
            print(name, parameter, parameter.shape)

    cnn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path.resolve()),
                       "code reserved space": code_size,
                       "n_inputs": 1,
                       "input_bits": 8,
                       "input_signed": True}

    with open(out_path.joinpath(f"config_{net_name}.json"), "w") as fp:
        json.dump(cnn_dory_config, fp, indent=4)

    #done!


def export_dvsnet(net_cnn : nn.Module, net_tcn : nn.Module, name : str, out_dir : str, eps_in : float, in_data : torch.Tensor, integerize : bool = True, D : float = 2**24, opset_version : int  = 10, change_n_levels : int = None, code_size : int = 310000):
    if isinstance(net_cnn, fx.GraphModule):
        cnn_window = get_input_channels(net_cnn)
    else:
        cnn_window = net_cnn.adapter.in_channels

    net_cnn = net_cnn.eval()
    net_tcn = net_tcn.eval()
    if change_n_levels:
        for m in chain(net_cnn.modules(), net_tcn.modules()):
            if isinstance(m, RequantShift):
                m.n_levels_out.data = torch.Tensor([change_n_levels])
    out_path_cnn = Path(out_dir).joinpath('cnn')
    out_path_tcn = Path(out_dir).joinpath('tcn')
    out_path_cnn.mkdir(parents=True, exist_ok=True)
    out_path_tcn.mkdir(parents=True, exist_ok=True)
    onnx_file_cnn = f"{name}_cnn_ql_integerized.onnx"
    onnx_file_tcn = f"{name}_tcn_ql_integerized.onnx"
    onnx_path_cnn = out_path_cnn.joinpath(onnx_file_cnn)
    onnx_path_tcn = out_path_tcn.joinpath(onnx_file_tcn)

    cnn_wins = torch.split(in_data, cnn_window, dim=1)
    tcn_window = len(cnn_wins)
    shape_in_cnn = cnn_wins[0].shape
    # atm no integerization is done here. Assume the nets are already integerized
    int_net_cnn = net_cnn
    int_net_tcn = net_tcn

    # the integerization pass annotates the conv layers with the number of
    # weight levels. from this information we can make a dictionary of the number of
    # weight bits.
    prec_dict_cnn = {}
    for lname, module in int_net_cnn.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            n_bits = int(np.log2(module.n_levels+1.2))
            prec_dict_cnn[lname] = n_bits
    prec_dict_tcn = {}
    for lname, module in int_net_tcn.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            n_bits = int(np.log2(module.n_levels+1.2))
            prec_dict_tcn[lname] = n_bits

    #first export an unannotated ONNX graph
    test_input_cnn = torch.rand(shape_in_cnn)
    torch.onnx.export(int_net_cnn.to('cpu'),
                      test_input_cnn,
                      str(onnx_path_cnn),
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True)

    int_net_cnn = int_net_cnn.to(torch.float64)
    #load the exported model and annotate it
    onnx_model_cnn = onnx.load(str(onnx_path_cnn))
    annotate_onnx(onnx_model_cnn, prec_dict_cnn)
    # finally, save the annotated ONNX model
    onnx.save(onnx_model_cnn, str(onnx_path_cnn))
    # now we pass a test input through the model and log the intermediate
    # activations

    # make a forward hook to dump outputs of RequantShift layers
    acts = []
    def dump_hook(self, inp, outp, lname):
        # DORY wants HWC tensors
        acts.append((lname, outp[0]))

    int_acts = []
    def dump_hook_dbg(self, inp, outp, lname):
        # DORY wants HWC tensors
        int_acts.append((lname, outp[0]))

    for n, m in int_net_cnn.named_modules():
        if isinstance(m, (RequantShift, nn.MaxPool1d, nn.MaxPool2d)):
            hook = partial(dump_hook, lname=n)
            m.register_forward_hook(hook)
    for n, m in int_net_cnn.named_modules():
        if isinstance(m, (nn.Conv2d)):
            hook = partial(dump_hook_dbg, lname=n)
            m.register_forward_hook(hook)
    for n, m in int_net_tcn.named_modules():
        if isinstance(m, (RequantShift, nn.MaxPool1d, nn.MaxPool2d, nn.Linear)):
            hook = partial(dump_hook, lname=n)
            m.register_forward_hook(hook)

    # save everything into beautiful text files
    def save_beautiful_text(t : torch.Tensor, layer_name : str, filename : str, out_path : Path):
        t = t.squeeze(0)
        if t.dim()==3:
            # expect a (C, H, W) tensor - DORY expects (H, W, C)
            t = t.permute(1,2,0)
        elif t.dim()==2:
            # expect a (C, D) tensor - DORY expects (D, C)
            t = t.permute(1,0)
        else:
            print(f"Not permuting output of layer {layer_name}...")

        filepath = out_path.joinpath(f"{filename}.txt")
        np.savetxt(str(filepath), t.detach().flatten().numpy().astype(np.int32), delimiter=',', header=f"# {layer_name} (shape {list(t.shape)}),", fmt="%1d,")
        #with open(str(filepath), 'w') as fp:
        #    fp.write(f"# {layer_name} (shape {list(t.shape)}),\n")
        #    for el in t.flatten():p
        #        fp.write(f"{int(el)},\n")

    # save the whole input tensor
    save_beautiful_text(in_data, "input", "input", out_path_cnn)
    cnn_outs = []
    # feed the windows one by one to the cnn
    for idx, cnn_win in enumerate(cnn_wins):
        cnn_win_out = int_net_cnn(cnn_win.to(dtype=torch.float64))
        cnn_outs.append(cnn_win_out)
        save_beautiful_text(cnn_win, f"input_{idx}", f"input_{idx}", out_path_cnn)
        save_beautiful_text(cnn_win_out, f"output_{idx}", f"output_{idx}", out_path_cnn)
        for jdx, (lname, t) in enumerate(acts):
            save_beautiful_text(t, lname, f"out_{idx}_layer{jdx}", out_path_cnn)
        acts = []
    cnn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path_cnn.resolve()),
                       "code reserved space": code_size,
                       "n_inputs": tcn_window,
                       "input_bits": 2,
                       "input_signed": True,
                       "input_shape": list(shape_in_cnn[-3:]),
                       "output_shape": list(cnn_outs[0].shape[-2:])}
    with open(out_path_cnn.joinpath(f"config_{name}_cnn.json"), "w") as fp:
        json.dump(cnn_dory_config, fp, indent=4)

    #first export an unannotated ONNX graph
    tcn_input = torch.stack(cnn_outs, dim=2)
    shape_in_tcn = tcn_input.shape
    test_input_tcn = torch.rand(shape_in_tcn)
    torch.onnx.export(int_net_tcn.to('cpu'),
                      test_input_tcn,
                      str(onnx_path_tcn),
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True)

    #load the exported model and annotate it
    onnx_model_tcn = onnx.load(str(onnx_path_tcn))
    annotate_onnx(onnx_model_tcn, prec_dict_tcn)
    # finally, save the annotated ONNX model
    onnx.save(onnx_model_tcn, str(onnx_path_tcn))
    int_net_tcn = int_net_tcn.to(torch.float64)
    int_acts = []
    acts = []
    output = int_net_tcn(tcn_input.to(dtype=torch.float64))

    save_beautiful_text(tcn_input, "input", "input", out_path_tcn)
    save_beautiful_text(output, "output", "output", out_path_tcn)
    for jdx, (lname, t) in enumerate(acts):
        save_beautiful_text(t, lname, f"out_layer{jdx}", out_path_tcn)


    int_net_tcn = int_net_tcn.to(torch.float64)

    print(f"tcn input shape: {tcn_input.shape}")
    tcn_dory_config = {"BNRelu_bits": 32,
                       "onnx_file": str(onnx_path_tcn.resolve()),
                       "code reserved space": code_size,
                       "n_inputs": 1,
                       "input_bits": 2,
                       "input_signed": False,
                       "input_shape": list(tcn_input.shape[-2:]),
                       "output_shape": output.shape[-1]}

    with open(out_path_tcn.joinpath(f"config_{name}_tcn.json"), "w") as fp:
        json.dump(tcn_dory_config, fp, indent=4)
    # now we pass a test input through the model and log the intermediate
    # activations
