/*
 * double_buffer.h
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



#ifndef __DOUBLEBUFFER_H__
#define __DOUBLEBUFFER_H__

#include "pmsis.h"

typedef struct DoubleBuffer {
    uint32_t addrs[2];
    int index;
} DoubleBuffer;

static inline void double_buffer_increment(DoubleBuffer * db) {
    db->index = (db->index + 1) % 2;
}

static inline uint32_t double_buffer_get_addr(DoubleBuffer db) {
    return db.addrs[db.index];
}

#endif // __DOUBLEBUFFER_H__
