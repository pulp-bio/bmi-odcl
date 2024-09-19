/*
 * net_utils.c
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


#include "net_utils.h"
#include "pmsis.h"

void print_perf(const char *name, const int cycles, const int macs) {
  float perf = (float) macs / cycles;
  printf("\n%s performance:\n", name);
  printf("  - num cycles: %d\n", cycles);
  printf("  - MACs: %d\n", macs );
  printf("  - MAC/cycle: %g\n", perf);
  printf("  - n. of Cores: %d\n\n", NUM_CORES);
}

void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true) {
  uint32_t sum = 0;
  for (int i = 0; i < size; i++) sum += d[i];

  printf("Checking %s: Checksum ", name);
  if (sum_true == sum)
    printf("OK\n");
  else
    printf("Failed: true [%u] vs. calculated [%u]\n", sum_true, sum);
}

