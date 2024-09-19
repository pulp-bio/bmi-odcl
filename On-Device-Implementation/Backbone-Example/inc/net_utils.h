/*
 * net_utils.h
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



#ifndef __PERF_UTILS_H__
#define __PERF_UTILS_H__
#include <stddef.h>
#include <stdint.h>

// Padding flags

#define NET_UTILS_PAD_TOP    (1 << 3)
#define NET_UTILS_PAD_RIGHT  (1 << 2)
#define NET_UTILS_PAD_BOTTOM (1 << 1)
#define NET_UTILS_PAD_LEFT   (1 << 0)
#define NET_UTILS_NO_PAD     (0)

typedef struct {
  unsigned int L3_input;
  unsigned int L3_output;
  unsigned int L3_after_weights;
  unsigned int L2_input;
  unsigned int bypass;
  unsigned int L2_output;
  unsigned int L2_weights;
  unsigned int L1_buffer;
  unsigned int ram;
  unsigned int padding;
  unsigned int layer_id;
} layer_args_t;

void print_perf(const char *name, const int cycles, const int macs);
void checksum(const char *name, const uint8_t *d, size_t size, uint32_t sum_true);
#endif
