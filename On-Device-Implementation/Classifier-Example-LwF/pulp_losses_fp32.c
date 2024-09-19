/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
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
 */

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia, Cristian Cioflan, Thorir Mar Ingolfsson, Lan Mei
*/ 

#include "math.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_losses_fp32.h"


void pulp_CrossEntropyLoss ( void * loss_args )
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;

  float loss = 0.0;
  for(int i=0; i<size; i++){
    loss += -target[i]*logf(outData[i]);
    
    #ifdef DEBUG
      printf("target: %f, out_diff: %f, out_data:%f\n", target[i], outDiff[i], outData[i]);
      printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);  
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;

  for(int i=0; i<size; i++){
    outDiff[i] = (-target[i]+outData[i]);
    
    #ifdef DEBUG
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }
}

void pulp_LwF_Loss ( void * loss_args )
{
  struct loss_args_LwF * args = (struct loss_args_LwF *) loss_args;
  float * outData = args->output->data;
  float * outData_div_T = args->output_div_T->data;
  float * outData_old = args->output_old->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;
  int * T_distill = args->T_distill;

  int T = *T_distill;
  float old_weights_sum = 0.0;

  float loss_ori = 0.0;
  for(int i=0; i<size; i++){
    loss_ori += -target[i]*logf(outData[i]);
    
    #ifdef DEBUG
      printf("target: %f, out_diff: %f, out_data:%f\n", target[i], outDiff[i], outData[i]);
      printf("loss:%f \n",loss);
    #endif
  }

  float loss_new = 0.0;
  for(int i=0; i<size; i++){
    loss_new += -(outData_old[i])*logf(outData_div_T[i]);
    
    #ifdef DEBUG
      printf("target: %f, out_diff: %f, out_data:%f\n", target[i], outDiff[i], outData[i]);
      printf("loss:%f \n",loss);
    #endif
  }

  float loss = loss_ori + loss_new;

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);  
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss_ori + loss_new;

  for(int i=0; i<size; i++){
    // outDiff[i] = (-target[i]+outData[i]);
    old_weights_sum += outData_old[i];
  }

  for(int i=0; i<size; i++){
    // outDiff[i] = (-target[i]+outData[i]);
    outDiff[i] = (-target[i]+outData[i]) - outData_old[i]/(float)T + 1/(float)T * outData_div_T[i] * old_weights_sum;

    #ifdef DEBUG
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }
}


void pulp_MSELoss ( void * loss_args ) 
{
  struct loss_args * args = (struct loss_args *) loss_args;
  float * outData = args->output->data;
  float * outDiff = args->output->diff;
  float * target = args->target;
  float * wr_loss = args->wr_loss;
  int size = args->output->dim;
  int off = 0;

  float loss = 0.0f;
  float meanval = 1.0f / size;
  
  #ifdef DEBUG
  printf("loss meanval is: %f\n", meanval);
  #endif
  
  for(int i=0; i<size; i++){
    loss += meanval * (target[i] - outData[i]) * (target[i] - outData[i]);

    #ifdef DEBUG
    printf("target: %f, out_diff: %f, out_data:%f\n", target[i], outDiff[i], outData[i]);
    printf("loss:%f \n",loss);
    #endif
  }

  // Skip printf profiling in debug mode
  #ifdef DEBUG
  #ifdef PROF_NET
  pi_perf_stop();
  #endif
  printf("\nLoss: %+.4f\n", loss);
  #ifdef PROF_NET
  pi_perf_start();
  #endif
  #endif  

  *wr_loss = loss;

  for(int i=0; i<size; i++){
    outDiff[i] = meanval * 2.0f *(outData[i] - target[i]);

    #ifdef DEBUG
    printf("target: %+.4f, out_diff: %+.4f, out_data:%+.4f\n", target[i], outDiff[i], outData[i]);
    #endif
  }

}