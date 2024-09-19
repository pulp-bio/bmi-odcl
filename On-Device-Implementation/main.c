/*
 * main.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Cristian Cioflan <cioflanc@iis.ee.ethz.ch>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
 * Lan Mei <lanmei@student.ethz.ch>
 *
 * Copyright (C) 2019-2024 ETH Zurich and University of Bologna
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


#define DEFINE_CONSTANTS
#include "mem.h"
#include "network.h"

#include "pmsis.h"
#include "net_utils.h"

// PULP-TrainLib
#include "net.h"

//#define VERBOSE 1

// Next steps: read labels from file / quantlab outputs, automize the workflow
int SAMPLE_LABELS[100] = {0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0};

unsigned int gpio_pin_measurement_id = 89;

void application(void * arg) {
  int start_timecount_full = 0;
/*
    Opening of Filesystem and Ram
*/
  mem_init();
  network_initialize();
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(500000);
  void *L2_FC_layer_weights_int8 = pi_l2_malloc(weights_size[6]);
  void *L2_FC_layer_weights_float;
  void *L2_FC_layer_biases_float;

  static float ce_loss;
  int predict_label;

  if (NULL == l2_buffer) {
#ifdef VERBOSE
    printf("ERROR: L2 buffer allocation failed.");
#endif
    pmsis_exit(-1);
  }
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_buffer);
#endif
  size_t l2_input_size = 15200;
  size_t input_size = 500000;
  int initial_dir = 1;

  // training
  #ifdef VERBOSE
  printf("\n-------------------------Training Phase-------------------------\n");
  #endif
  int num_epochs = 1;
  int num_train_samples = 5;
  int num_val_samples = 5;

  void *ram_input = ram_malloc(input_size);

  L2_FC_layer_weights_float = pi_l2_malloc(2 * 928 * 4); // 4 bytes per float
  if (L2_FC_layer_weights_float == NULL) {
      printf("failed to allocate memory for L2_FC_layer_weights_float\n");
  }

  L2_FC_layer_biases_float = pi_l2_malloc(2 * 4); // 4 bytes per float
  if (L2_FC_layer_biases_float == NULL) {
      printf("failed to allocate memory for L2_FC_layer_biases_float\n");
  }

  for (int idx_epochs = 0; idx_epochs < num_epochs; idx_epochs++){
    #ifdef VERBOSE
    printf("\nCur training EPOCH: %d\n", idx_epochs+1);
    #endif
    int correct_count_train = 0;
    pi_gpio_pin_write(gpio_pin_measurement_id, 1);
    for (int i = 0; i < num_train_samples; i++) {
      int start_timecount = 0;
      //if (i == 0) 
        start_timecount = pi_time_get_us();
      #ifdef VERBOSE
      printf("\nCur training input idx: %d\n", i);
      printf("-----------------------Run DORY Network except FC-----------------------\n");
      #endif
      load_file_to_ram(ram_input, Input_names[i]);  // "inputs.hex", Input_names[i]
      ram_read(l2_buffer, ram_input, l2_input_size);
      network_run(l2_buffer, 500000, l2_buffer, L2_FC_layer_weights_int8, i, initial_dir);
      //checksum("L2_FC_layer_weights_int8 weights new out", L2_FC_layer_weights_int8, weights_size[6], weights_checksum[6]);
      //checksum("final output",l2_buffer, activations_out_size[5], activations_out_checksum[5][0]);

      //ram_free(ram_input, input_size);
      //network_terminate();

      #ifdef VERBOSE
      printf("-----------------------Run PULP-TrainLib FC-----------------------\n");
      #endif
      //printf("\nHello there.\nConfiguring cluster..\n");
      // Configure cluster
      struct pi_device cluster_dev;
      struct pi_cluster_conf cl_conf;
      struct pi_cluster_task cl_task;

      pi_cluster_conf_init(&cl_conf);
      pi_open_from_conf(&cluster_dev, &cl_conf);
      if (pi_cluster_open(&cluster_dev))
      {
          return -1;
      }

      unsigned int args_init[6];
      args_init[0] = (unsigned int) l2_buffer;
      args_init[1] = (unsigned int) L2_FC_layer_weights_int8;
      args_init[2] = (unsigned int) L2_FC_layer_weights_float;
      args_init[3] = (unsigned int) 1; // update = 0: no update (validate), 1: update (train).
      if (i == 0)
        args_init[4] = (unsigned int) 1; // init = 1
      else
        args_init[4] = (unsigned int) 0; // init = 0
      args_init[5] = (unsigned int) SAMPLE_LABELS[i]; // sample labels
      args_init[6] = (float*) &ce_loss;
      args_init[7] = (int*) &predict_label;
      args_init[8] = (unsigned int) L2_FC_layer_biases_float;

      //printf("\nLaunching training procedure...\n");
      pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, net_step, args_init));

      #ifdef VERBOSE
      printf("Current CE Loss: %f!\n", ce_loss);
      printf("Current Prediction: %d!\n", predict_label);
      printf("Current label: %d!\n", SAMPLE_LABELS[i]);
      #endif
      if (predict_label == SAMPLE_LABELS[i])  correct_count_train++;
      pi_cluster_close(&cluster_dev);

      //if (i == 0) {
        int end_timecount = pi_time_get_us();
        printf("Train One Sample: %i us\n", end_timecount - start_timecount);
      //}
    }

    pi_gpio_pin_write(gpio_pin_measurement_id, 0);

    float train_acc = (float) correct_count_train / num_train_samples;

    #ifdef VERBOSE
    printf("Net training done!\n");
    printf("Current Training Acc: %f!\n", train_acc);
    #endif
  }

  int correct_count_val = 0;

  // validation
  #ifdef VERBOSE
  printf("\n-------------------------Validation Phase-------------------------\n");
  #endif

  for (int i = 0; i < num_val_samples; i++) {
    int start_timecount = 0;
    //if (i == 0) 
      start_timecount = pi_time_get_us();
    #ifdef VERBOSE
    printf("\nCur validation input idx: %d\n", i);
    printf("-----------------------Run DORY Network except FC-----------------------\n");
    #endif
    //void *ram_input = ram_malloc(input_size); // Note the sequence here!! Cannot be put here in the loop, otherwise will read wrong input data!
    load_file_to_ram(ram_input, Input_names[i]);  // "inputs.hex", Input_names[i]
    ram_read(l2_buffer, ram_input, l2_input_size);
    network_run(l2_buffer, 500000, l2_buffer, L2_FC_layer_weights_int8, i, initial_dir);
    //checksum("L2_FC_layer_weights_int8 weights new out", L2_FC_layer_weights_int8, weights_size[6], weights_checksum[6]);
    //checksum("final output",l2_buffer, activations_out_size[5], activations_out_checksum[5][0]);

    #ifdef VERBOSE
    printf("-----------------------Run PULP-TrainLib FC-----------------------\n");
    #endif
    // Configure cluster
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    struct pi_cluster_task cl_task;

    pi_cluster_conf_init(&cl_conf);
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        return -1;
    }

    unsigned int args_init[6];
    args_init[0] = (unsigned int) l2_buffer;
    args_init[1] = (unsigned int) L2_FC_layer_weights_int8;
    args_init[2] = (unsigned int) L2_FC_layer_weights_float;
    args_init[3] = (unsigned int) 0; // update = 0: no update (validate), 1: update (train).
    args_init[4] = (unsigned int) 0; // init = 0
    args_init[5] = (unsigned int) SAMPLE_LABELS[i]; // dummy class
    args_init[6] = (float*) &ce_loss;
    args_init[7] = (int*) &predict_label;
    args_init[8] = (unsigned int) L2_FC_layer_biases_float;

    //printf("\nLaunching training procedure...\n");
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, net_step, args_init));

    #ifdef VERBOSE
    printf("Current CE Loss: %f!\n", ce_loss);
    printf("Current Prediction: %d!\n", predict_label);
    printf("Current label: %d!\n", SAMPLE_LABELS[i]);
    #endif
    if (predict_label == SAMPLE_LABELS[i])  correct_count_val++;
    pi_cluster_close(&cluster_dev);

    //if (i == 0) {
      int end_timecount = pi_time_get_us();
      printf("Validate One Sample: %i us\n", end_timecount - start_timecount);
    //}
  }

  float val_acc = (float) correct_count_val / num_val_samples;

  #ifdef VERBOSE
  printf("Net validation done!\n");
  printf("Current Validation Acc: %f!\n", val_acc);
  #endif
  
  ram_free(ram_input, input_size);
  network_terminate();

  pi_l2_free(l2_buffer, 500000);

  pi_l2_free(L2_FC_layer_weights_float, 2 * 928 * 4);

  pi_l2_free(L2_FC_layer_biases_float, 2 * 4);

  pi_l2_free(L2_FC_layer_weights_int8, weights_size[6]);

  int end_timecount_full = pi_time_get_us();
  #ifdef VERBOSE
  printf("Full time: %i us\n", end_timecount_full - start_timecount_full);
  #endif

  pmsis_exit(0);
}

int main () {
#ifndef TARGET_CHIP_FAMILY_GAP9
  PMU_set_voltage(1000, 0);
#else
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, 800);
#endif
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 370000000); // 370000000
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 370000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 370000000);
  pi_time_wait_us(10000);

  pi_pad_function_set(gpio_pin_measurement_id, 1);
  pi_gpio_pin_configure(gpio_pin_measurement_id, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(gpio_pin_measurement_id, 0);

  pmsis_kickoff((void*)application);
  pi_gpio_pin_write(gpio_pin_measurement_id, 0);

  return 0;
}
