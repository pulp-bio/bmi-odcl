/*
 * test_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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
#include "mem.h"
#include "network.h"

#include "pmsis.h"

#define VERBOSE 1



void application(void * arg) {
/*
    Opening of Filesystem and Ram
*/
  mem_init();
  network_initialize();
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(1386000);
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
  size_t input_size = 1000000;
  int initial_dir = 1;

  void *ram_input = ram_malloc(input_size);
  for (int exec = 0; exec < 100; exec++) {
      load_file_to_ram(ram_input, Input_names[exec]);
      ram_read(l2_buffer, ram_input, l2_input_size);
      network_run(l2_buffer, 1386000, l2_buffer, exec, initial_dir);

  }
  ram_free(ram_input, input_size);
  network_terminate();
  pi_l2_free(l2_buffer, 1386000);
}

int main () {

/*
#ifndef TARGET_CHIP_FAMILY_GAP9
  PMU_set_voltage(1000, 0);
#else
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, PI_PMU_VOLT_800);
#endif
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 370000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 370000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 370000000);
  pi_time_wait_us(10000);
*/

  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 240*1000*1000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 240*1000*1000);
  pi_time_wait_us(10000);
  uint32_t voltage = 650;

  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, voltage);
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, voltage);


  pmsis_kickoff((void*)application);
  return 0;
}
