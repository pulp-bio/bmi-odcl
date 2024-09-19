/*
 * tile_index.h
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



#ifndef __TILE_INDEX_H__
#define __TILE_INDEX_H__

typedef struct TileIndex {
    int height, width, input_channel, output_channel;
} TileIndex;

/*
 * Index order:
 * output_channel -> height -> width -> input_channel
 */
static TileIndex tile_index_get_next(TileIndex index, TileIndex end) {
    index.input_channel += 1;
    if (index.input_channel >= end.input_channel) {
        index.input_channel = 0;
        index.width += 1;
        if (index.width >= end.width) {
            index.width = 0;
            index.height += 1;
            if (index.height >= end.height) {
                index.height = 0;
                index.output_channel += 1;
                if (index.output_channel >= end.output_channel) {
                    index.output_channel = 0;
                }
            }
        }
    }
    return index;
}

/*
 * Index order:
 * input_channel, output_channel -> height -> width
 */
static TileIndex tile_index_get_next_dw(TileIndex index, TileIndex end) {
    index.width += 1;
    if (index.width >= end.width) {
        index.width = 0;
        index.height += 1;
        if (index.height >= end.height) {
            index.height = 0;
            index.output_channel += 1;
            index.input_channel += 1;
            if (index.output_channel >= end.output_channel) {
                index.output_channel = 0;
            }
            if (index.input_channel >= end.input_channel) {
                index.input_channel = 0;
            }
        }
    }
    return index;
}

#endif // __TILE_INDEX_H__
