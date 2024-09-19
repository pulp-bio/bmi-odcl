'''
Copyright (C) 2021-2024 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini, Leonardo Ravaglia, Cristian Cioflan, Thorir Mar Ingolfsson, Lan Mei
'''


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import dump_utils as dump
import numpy as np

#Visualize data with more precision
torch.set_printoptions(precision=10, sci_mode=False)

parser = argparse.ArgumentParser("FCL Layer Test")
parser.add_argument( '--in_size', type=int, default=928 )
parser.add_argument( '--out_size', type=int, default=2 )
parser.add_argument( '--file_name', type=str, default='linear-data.h')
parser.add_argument( '--step', type=str, default='FORWARD_BACKWARD_PROP')     # Possible steps: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=1)
args = parser.parse_args()

# Network parametersin_size
lr = args.lr
num_epochs = args.num_epochs
in_size = args.in_size
out_size = args.out_size
simple_kernel = False
current_step = args.step

# Net step
f_step = open('step-check.h', 'w')
f_step.write('#define ' + str(current_step) + '\n')
f_step.close()

# Data file
f = open(args.file_name, "w") 

f.write('#define LEARNING_RATE ' + str(lr) + 'f\n' )
f.write('#define NUM_EPOCHS ' + str(num_epochs) + '\n')

f.write('#define Tin_l0 ' + str(in_size) + '\n')
f.write('#define Tout_l0 ' + str(out_size) + '\n\n')

f.write("#define L0_IN_CH     (Tin_l0)\n")
f.write("#define L0_OUT_CH    (Tout_l0)\n")
f.write("#define L0_WEIGHTS   (L0_IN_CH*L0_OUT_CH)\n")

# Sample linear layer
class LinLayer (nn.Module):

    def __init__(self):
        super(LinLayer, self).__init__()
        self.lin = nn.Linear(in_features=in_size, out_features=out_size, bias=True)

    def forward(self, x):
        out = self.lin(x)
        return out


## Training hyperparameters, with manual initializations
#lr = 1
#initial_weights = torch.zeros(out_size, in_size) 

#temp_value = 0.01
#if simple_kernel:
#    initial_weights[0:out_size] = 0.01
#else:
#    for i in range(out_size):
#        for j in range(in_size):
#            initial_weights[i][j] = temp_value
#            temp_value = temp_value + 0.01

eps_in = 0.0039215689
eps_w = 0.0010615624

# Loading weights and input data from saved npy files in exported folder from quantlab
initial_weights = torch.from_numpy(np.squeeze(np.load("weights_fc.npy"))).to(torch.float32)
initial_weights = torch.reshape((torch.transpose(torch.reshape(initial_weights, (2,32,29)), 1, 2)), (2,928))
sample_labels = np.load(f"inputs/inputs_fc_labels.npy")
for i in range(out_size):
    for j in range(in_size):
        initial_weights[i][j] = initial_weights[i][j] * eps_w

initial_bias = torch.from_numpy(np.squeeze(np.load("bias_fc.npy"))).to(torch.float32)
for i in range(out_size):
    initial_bias[i] = initial_bias[i] * eps_w * eps_in

# Define and initialize net
net = LinLayer()
#print("\nInitializing net parameters to {}.\nParameters are: ".format(initial_weights))


net.lin.weight = nn.Parameter(initial_weights)
net.lin.bias = nn.Parameter(initial_bias)
for name, parameter in net.named_parameters():
    print(name, parameter, parameter.shape)

net.zero_grad()

# Optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0)

# training process
num_train_inputs = 100
net.train()

#num_epochs = 1

for i_epochs in range(num_epochs):

    for input_idx in range(num_train_inputs):
        print("\nCur training input idx is: ", input_idx)
        indata = torch.from_numpy(np.squeeze(np.load(f"inputs/inputs_fc_{input_idx}.npy"))).to(torch.float32)
        indata = torch.flatten(torch.transpose(torch.reshape(indata, (32,29)), 0, 1)) 
        for i in range(in_size):
            indata[i] = indata[i] * eps_in
        indata.requires_grad = True

        if input_idx == 0:
            f.write('PI_L2 float INPUT_VECTOR[L0_IN_CH] = {'+dump.tensor_to_string(indata)+'};\n')

        if input_idx == 0:
            f.write('PI_L2 float L0_WEIGHTS_params[L0_WEIGHTS] = {'+dump.tensor_to_string(net.lin.weight)+'};\n')

        label = torch.zeros(1, out_size)
        label[0][sample_labels[input_idx]] = 1 

        print("\nLabel is: ", label, label.shape, label.dtype)
        if input_idx == 0:
            f.write('PI_L2 float LABEL[L0_OUT_CH] = {'+dump.tensor_to_string(label)+'};\n')

        # Do a forward computation
        #optimizer.zero_grad()
        #indata.grad = None
        print("\nCur net bias is: ", net.lin.bias)
        optimizer.zero_grad()
        output = net(indata).unsqueeze(dim=0)
        # print("\nNet output is: ", output, output.shape, output.dtype)
        # f.write('PI_L2 float L0_OUT_FW [L0_OUT_CH] = {'+dump.tensor_to_string(output)+'};\n')
        # print(output)
        # print(label)
        loss = criterion(output, label)
        # print("\nLoss is: ", loss, loss.shape, loss.dtype)
        # f.write('PI_L2 float L0_LOSS = '+str(loss.item())+';\n')
        # Manually compute outdiff
        # loss_meanval = 1/out_size
        # output_diff = loss_meanval * 2.0 * (output - label)
        # print("\nOutput loss is: ", output_diff, output_diff.shape, output_diff.dtype)
        # f.write('PI_L2 float L0_OUT_GRAD [L0_OUT_CH] = {'+dump.tensor_to_string(output_diff)+'};\n')
        
        loss.backward()
        #output_diff = -label + output
        optimizer.step()
        # ("\nNetwork gradients are: ")
        # for name, parameter in net.named_parameters():
        #     print(name, parameter.grad, parameter.grad.shape, parameter.grad.dtype)
        # f.write('PI_L2 float L0_WEIGHT_GRAD [L0_WEIGHTS] = {'+dump.tensor_to_string(parameter.grad)+'};\n')

        # print("\nInput grad is: ", indata.grad)
        # f.write('PI_L2 float L0_IN_GRAD [L0_IN_CH] = {'+dump.tensor_to_string(indata.grad)+'};\n')

        # f.write('\n\n')

        #print("\nLoss is: ", loss, loss.shape, loss.dtype)
        
        if i_epochs == num_epochs - 1:
        
            print("\nNet output is: ", output, output.shape, output.dtype)
            if input_idx == 0:
                f.write('PI_L2 float L0_OUT_FW_LAST [L0_OUT_CH] = {'+dump.tensor_to_string(output)+'};\n')
            print("\nLoss is: ", loss, loss.shape, loss.dtype)
            if input_idx == 0:
                f.write('PI_L2 float L0_LOSS_LAST = '+str(loss.item())+';\n')
            #print("\nOutput loss is: ", output_diff, output_diff.shape, output_diff.dtype)
            #f.write('PI_L2 float L0_OUT_GRAD [L0_OUT_CH] = {'+dump.tensor_to_string(output_diff)+'};\n')
            print("\nNetwork gradients are: ")
            for name, parameter in net.named_parameters():
                print(name, parameter.grad, parameter.grad.shape, parameter.grad.dtype)
                if name == 'lin.weight':
                    if input_idx == 0:
                        f.write('PI_L2 float L0_WEIGHT_GRAD_LAST [L0_WEIGHTS] = {'+dump.tensor_to_string(parameter.grad)+'};\n')
                elif name == 'lin.bias': 
                    if input_idx == 0:
                        f.write('PI_L2 float L0_BIAS_GRAD_LAST [L0_OUT_CH] = {'+dump.tensor_to_string(parameter.grad)+'};\n')
            #print("\nInput grad is: ", indata.grad)
            #f.write('PI_L2 float L0_IN_GRAD_LAST [L0_IN_CH] = {'+dump.tensor_to_string(indata.grad)+'};\n')

            if input_idx == 0:
                f.write('\n\n')

            print("\nLoss is: ", loss, loss.shape, loss.dtype)

#print(sample_labels)
f.close()

net.eval()
num_val_inputs = 100
num_correct = 0

for input_idx in range(num_val_inputs):
    print("\nCur validation input idx is: ", input_idx)
    indata = torch.from_numpy(np.squeeze(np.load(f"inputs/inputs_fc_{input_idx}.npy"))).to(torch.float32)
    indata = torch.flatten(torch.transpose(torch.reshape(indata, (32,29)), 0, 1)) 
    for i in range(in_size):
        indata[i] = indata[i] * eps_in
    #indata.requires_grad = True

    #indata = torch.div(torch.ones(in_size), 100000)
    #indata.requires_grad = True
    #print("\nInput data is: ", indata, indata.shape, indata.dtype)

    cur_label = sample_labels[input_idx]

    label = torch.zeros(1, out_size)
    label[0][sample_labels[input_idx]] = 1 

    print("\nValidation Label is: ", cur_label)

    net.eval()
    output_predict = net(indata)
    output = output_predict.unsqueeze(dim=0)
    # print("\nNet output is: ", output, output.shape, output.dtype)
    # print(output)

    loss = criterion(output, label)

    predicted_label = np.argmax(output_predict.detach().numpy())
    if cur_label == predicted_label:
        num_correct += 1
    
    print("\nNet output is: ", output, output.shape, output.dtype)
    print("\nLoss is: ", loss, loss.shape, loss.dtype)

val_acc = num_correct / num_val_inputs
print("\nValidation accuracy is: ", val_acc)