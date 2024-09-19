/*
 * Copyright (C) 2021-2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 */

// Authors: Davide Nadalini, Leonardo Ravaglia, Cristian Cioflan, Thorir Mar Ingolfsson, Lan Mei

#define MEMOCC_COMP

#include "pulp_train.h"

#include "linear-data.h"
#include "stats.h"

#include "net.h"

// DATA DEFINITION


PI_L2 float L0_BIAS_params[Tout_l0];

// LINEAR
PI_L2 struct Linear_args FC_args; //PI_L1
PI_L2 struct act_args act_args; //PI_L1
PI_L2 struct blob layer0_in, layer0_wgt, layer0_out, layer0_act_out, layer0_bias; //PI_L1
// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

PI_L1 float zero_init = 0.0f; //PI_L1


#ifdef FORWARD_BACKWARD_PROP
PI_L2 float l0_in[Tin_l0]; //PI_L1
PI_L2 float l0_ker[Tker_l0]; //PI_L1
PI_L2 float l0_out[Tout_l0]; //PI_L1
PI_L2 float l0_in_diff [Tin_l0]; //PI_L1
PI_L2 float l0_ker_diff[Tker_l0]; //PI_L1
PI_L2 float l0_out_diff [Tout_l0]; //PI_L1
PI_L2 float l0_act_out[Tout_l0]; //PI_L1
PI_L2 float l0_act_out_diff [Tout_l0]; //PI_L1

PI_L2 float l0_bias[Tout_l0]; //PI_L1
PI_L2 float l0_bias_diff[Tout_l0]; //PI_L1

PI_L2 float l0_final_out[Tout_l0]; //PI_L1


PI_L2 float loss = 0; //PI_L1
PI_L2 struct loss_args loss_args; //PI_L1
#endif

#ifdef FORWARD_BACKWARD_PROP
static inline void tensor_init() 
{
  for (int i=0; i<Tin_l0; i++)        l0_in[i] = INPUT_VECTOR[i];
  for (int i=0; i<Tker_l0; i++)       l0_ker[i] = L0_WEIGHTS_params[i];
  for (int i=0; i<Tout_l0; i++)       l0_out[i] = zero_init; 
  
  for (int i=0; i<Tin_l0; i++)        l0_in_diff[i] = zero_init;
  for (int i=0; i<Tker_l0; i++)       l0_ker_diff[i] = zero_init;
  for (int i=0; i<Tout_l0; i++)       l0_out_diff[i] = zero_init;  

  for (int i=0; i<Tout_l0; i++)       l0_act_out[i] = zero_init;
  for (int i=0; i<Tout_l0; i++)       l0_act_out_diff[i] = zero_init;
  
}

static inline void tensor_init_from_DORY() 
{
  for (int i=0; i<Tin_l0; i++)        l0_in[i] = INPUT_VECTOR[i];
  for (int i=0; i<Tker_l0; i++)       l0_ker[i] = L0_WEIGHTS_params[i];
  for (int i=0; i<Tout_l0; i++)       l0_out[i] = zero_init; 
  
  for (int i=0; i<Tin_l0; i++)        l0_in_diff[i] = zero_init;
  for (int i=0; i<Tker_l0; i++)       l0_ker_diff[i] = zero_init;
  for (int i=0; i<Tout_l0; i++)       l0_out_diff[i] = zero_init;  

  for (int i=0; i<Tout_l0; i++)       l0_act_out[i] = zero_init;
  for (int i=0; i<Tout_l0; i++)       l0_act_out_diff[i] = zero_init;

  for (int i=0; i<Tout_l0; i++)       l0_bias[i] = L0_BIAS_params[i];
  for (int i=0; i<Tout_l0; i++)       l0_bias_diff[i] = zero_init;  
  
}

static inline void connect_blobs() 
{
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_l0;
  layer0_in.diff = l0_in_diff;

  layer0_wgt.data = l0_ker;
  layer0_wgt.dim = Tker_l0;
  layer0_wgt.diff = l0_ker_diff;

  layer0_bias.data = l0_bias;
  layer0_bias.dim = Tout_l0;
  layer0_bias.diff = l0_bias_diff;
  
  layer0_out.data = l0_out;
  layer0_out.dim = Tout_l0;
  layer0_out.diff = l0_out_diff;
  
  layer0_act_out.data = l0_act_out;
  layer0_act_out.dim = Tout_l0;
  layer0_act_out.diff = l0_act_out_diff;

  FC_args.input = &layer0_in;
  FC_args.coeff = &layer0_wgt;
  FC_args.output = &layer0_out;
  FC_args.skip_in_grad = 0;
  FC_args.opt_matmul_type_fw = MATMUL_TYPE;
  FC_args.opt_matmul_type_wg = MATMUL_TYPE;
  FC_args.opt_matmul_type_ig = MATMUL_TYPE;
  
  act_args.input = &layer0_out;
  act_args.output = &layer0_act_out;
}

static inline void compute_memory_occupation(){
  // Input
  L1_memocc_bytes += Tin_l0*sizeof(float);
  // Kernel
  L1_memocc_bytes += Tker_l0*sizeof(float); 
  // Output
  L1_memocc_bytes += Tout_l0*sizeof(float);
  
  // Input gradient
  L1_memocc_bytes += Tin_l0*sizeof(float);
  // Kernel gradient
  L1_memocc_bytes += Tker_l0*sizeof(float); 
  // Output gradient
  //L1_memocc_bytes += Tout_l0*sizeof(float);

  // Input data
  L2_memocc_bytes += L0_IN_CH*sizeof(float);
  // Weights
  L2_memocc_bytes += L0_WEIGHTS*sizeof(float);
  // Output
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);
  // Output gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);
  // Weight gradient
  L2_memocc_bytes += L0_WEIGHTS*sizeof(float);
  // Input gradient
  //L2_memocc_bytes += L0_IN_CH*sizeof(float);
}

static inline void update_weights_and_bias()
{
  struct optim_args opt_l0;
  opt_l0.weights = &layer0_wgt;
  opt_l0.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0);
  // pulp_gradient_descent_fp32(&opt_l0);

  // gradient descent - bias!
  struct optim_args opt_l0_bias;
  opt_l0_bias.weights = &layer0_bias;
  opt_l0_bias.learning_rate = LEARNING_RATE;
  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l0_bias);
}
#endif

static inline void compare_tensors(float *A, float *B, int length){

  float mean_err_rel = 0.0f;
  float diff = 0.0f;
  float den = 0.000001f; //0.000001f;

  for(int i=0; i<length; i++){
     if (A[i]>B[i] && A[i]>0.0f){
        diff = A[i]-B[i];
        if (diff>0) diff = diff;
        else diff=-diff;
        if (A[i]>0) den = A[i];
        //else den = -A[i];
        else if (A[i]<0) den = -A[i]; // missing A = 0
        else den = 0.000001f;
        mean_err_rel = mean_err_rel + (diff / den)/length;
     }
     else{
       diff = A[i]-B[i];
       if (diff>0) diff = diff;
       else diff=-diff;
       if (A[i]>0) den = A[i];
       //else den = -A[i];
       else if (A[i]<0) den = -A[i]; // missing A = 0
       else den = 0.000001f;
       mean_err_rel = mean_err_rel + (diff / den)/length;
     }
  }
  if (mean_err_rel<ERROR_TOLERANCE) printf(">>>TENSOR MATCHING!\n");
  else printf(">>>TENSOR NOT MATCHING!\n");

}

// Elementwise checker
int check_tensor(float * tensor_out, float * tensor_ref, int size){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



static inline void train(){

  #ifdef FORWARD_BACKWARD_PROP
  pulp_linear_fp32_fw_cl(&FC_args);

  for (int i=0; i<Tout_l0; i++)       l0_out[i] = l0_out[i] + l0_bias[i]; 

  pulp_softmax_fp32_fw_cl(&act_args);
  
  loss_args.output = &layer0_act_out;
  loss_args.target = LABEL;
  loss_args.wr_loss = &loss;
  // NOTE: Davide modified outDiff[i] = (-target[i]+outData[i]) to outDiff[i] = (-target[i] / outData[i]) in a new commit
  pulp_CrossEntropyLoss(&loss_args);
  for (int i=0; i<Tout_l0; i++)       l0_out_diff[i] = layer0_act_out.diff[i]; 

  //for (int i=0; i<Tin_l0; i++) {
  //  printf("INPUT: %f    \n", FC_args.input->data[i]); //((float) *((uint8_t *l2_buffer)+i)) * eps_in;
  //  //printf("%d     %f    \n", ((uint8_t*)l2_buffer)[i], INPUT_VECTOR[i]);
  //}
  
  pulp_linear_fp32_bw_param_grads_cl(&FC_args);

  // Derivative of bias can directly be regarded as derivative of outputs, as y = w*x + b, derivative of b = 1.
  for (int i=0; i<Tout_l0; i++)       l0_bias_diff[i] = layer0_act_out.diff[i]; 

  #endif
}


// Most important function: it connects each passage to step the net and perform training
void net_step(void *args)
{

  unsigned int * real_args = (unsigned int *) args;
  void * l2_buffer = (void *) real_args[0];
  void * L2_FC_layer_weights_int8 = (void *) real_args[1];
  void * L2_FC_layer_weights_float = (void *) real_args[2];
  int update = (int) real_args[3];
  int init = (int) real_args[4];
  int class = (int) real_args[5];
  float * ce_loss = (float *) real_args[6];
  int * predict_label = (int *) real_args[7];
  void * L2_FC_layer_biases_float = (void *) real_args[8];

  #ifdef PROF_NET
  INIT_STATS();
  PRE_START_STATS();
  START_STATS();
  #endif
  
  #ifdef VERBOSE
  #ifdef FORWARD_BACKWARD_PROP
  printf("Forward backward prop:\n");
  printf("epochs = %d, lr = %f", NUM_EPOCHS, LEARNING_RATE);
  #endif

  #ifdef MEMOCC_COMP
  if (init == 1)
    compute_memory_occupation();
  printf("\nL1 memory occupation: %d bytes.", L1_memocc_bytes);
  printf("\nL2 memory occupation: %d bytes.\n", L2_memocc_bytes);
  #endif
  #endif

  // Float32 inputs
  float eps_in = 0.0039215689; // update eps per dataset!

  for (int i=0; i<Tin_l0; i++) {
    INPUT_VECTOR[i] = ((float)((uint8_t*)l2_buffer)[i]) * eps_in; 
  }

  // Float32 weights
  float eps_w = 0.0010615624; // update eps per dataset!

  if (init == 1) {
    for (int i=0; i<Tin_l0*Tout_l0; i++) {
      L0_WEIGHTS_params[i] = ((float)((int8_t*)L2_FC_layer_weights_int8)[i]) * eps_w;
      // printf("%d     %f    \n", ((int8_t*)L2_FC_layer_weights_int8)[i], L0_WEIGHTS_params[i]);
    }
    #ifdef VERBOSE
    printf("Initialized weights! \n");
    #endif
  } else {
    for (int i=0; i<Tin_l0*Tout_l0; i++) {
      //printf("%f    \n", ((float*)L2_FC_layer_weights_float)[i]);
      L0_WEIGHTS_params[i] = ((float*)L2_FC_layer_weights_float)[i];
    }
    #ifdef VERBOSE
    printf("Copied weights! \n");
    #endif
  }

  float eps_b = eps_in * eps_w;

  int32_t *L2_FC_layer_bias = (int8_t*)L2_FC_layer_weights_int8 + Tin_l0*Tout_l0;


  if (init == 1) {
    for (int i=0; i<Tout_l0; i++) {
      L0_BIAS_params[i] = ((float) L2_FC_layer_bias[i]) * eps_b; 
      //printf("Bias[%d]: %f    \n", i, L0_BIAS_params[i]);
    }
    #ifdef VERBOSE
    printf("Initialized biases! \n");
    #endif
  } else {
    for (int i=0; i<Tout_l0; i++) {
      //printf("%f    \n", ((float*)L2_FC_layer_biases_float)[i]);
      L0_BIAS_params[i] = ((float*)L2_FC_layer_biases_float)[i];
    }
    #ifdef VERBOSE
    printf("Copied biases! \n");
    #endif
  }

  for (int i=0; i<Tout_l0; i++){
      LABEL[i] = 0.0f;
  } 
  LABEL[class] = 1.0f;

  //tensor_init();
  tensor_init_from_DORY();
  connect_blobs();
  
  #ifdef FORWARD_BACKWARD_PROP

  for (int epoch=0; epoch < NUM_EPOCHS; epoch++){
    train();
    if (update == 1)
      update_weights_and_bias();
  }
  
  #endif
  
  #ifdef PROF_NET
  STOP_STATS();
  #endif


  #ifdef VERBOSE
  if (init == 1) {
    // TODO: Also implement checks for samples with idx > 0
    #ifdef FORWARD_BACKWARD_PROP
    printf("FORWARD CHECK: \n");
    compare_tensors(l0_out, L0_OUT_FW_LAST, Tout_l0);
    check_tensor(l0_out, L0_OUT_FW_LAST, Tout_l0);

    // //printf("OUT GRADIENT CHECK: \n");
    // //compare_tensors(l0_out_diff, L0_OUT_GRAD, Tout_l0);
    // //check_tensor(l0_out_diff, L0_OUT_GRAD, Tout_l0);

    printf("WEIGHTS GRADIENT CHECK: \n"); 
    compare_tensors(l0_ker_diff, L0_WEIGHT_GRAD_LAST, Tker_l0);
    check_tensor(l0_ker_diff, L0_WEIGHT_GRAD_LAST, Tker_l0);

    printf("BIAS GRADIENT CHECK: \n"); 
    compare_tensors(l0_bias_diff, L0_BIAS_GRAD_LAST, Tout_l0);
    check_tensor(l0_bias_diff, L0_BIAS_GRAD_LAST, Tout_l0);

    // //printf("INPUTS GRADIENT CHECK: \n"); 
    // //compare_tensors(l0_in_diff, L0_IN_GRAD_LAST, Tin_l0);
    // //check_tensor(l0_in_diff, L0_IN_GRAD_LAST, Tin_l0);
    #endif

  } else {
    printf("FORWARD CHECK: \n");
    compare_tensors(l0_out, L0_OUT_FW_LAST, Tout_l0);
    check_tensor(l0_out, L0_OUT_FW_LAST, Tout_l0);
  }

  printf("LOSS CHECK - Last epoch: \n"); 
  printf("Real loss is %f, GM loss is %f: \n", loss, L0_LOSS_LAST); 
  //printf("Real loss is %f\n", loss); 
  #endif

  // if (init == 1) {
  if (update == 1) {
    for (int i=0; i<Tin_l0*Tout_l0; i++){
      ((float*)L2_FC_layer_weights_float)[i] = layer0_wgt.data[i];
      //printf("%f    \n", ((float*)L2_FC_layer_weights_float)[i]);
    }
    for (int i=0; i<Tout_l0; i++) {
      ((float*)L2_FC_layer_biases_float)[i] = layer0_bias.data[i];
    }
    #ifdef VERBOSE
    printf("Weights updated and stored in L2_FC_layer_weights_float!\n");
    #endif
  }

  float max_val = -1E-37;
  int max_idx = -1;
  for (int i=0; i<Tout_l0; i++) {
    if (l0_out[i] > max_val) {
      max_val = l0_out[i];
      max_idx = i;
    }
  }

  *predict_label = max_idx;
  //printf("\n Current prediction: %d \n", *predict_label);

  *ce_loss = loss;

  return;
}
