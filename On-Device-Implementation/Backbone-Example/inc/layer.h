/*
 * layer.h
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



#ifndef __LAYER_H__
#define __LAYER_H__

#include "pmsis.h"
#include "tile_index.h"

typedef struct Address {
    uint32_t input;
    uint32_t weights;
    uint32_t scale;
    uint32_t bias;
    uint32_t output;
} Address;

typedef struct Padding {
    int top, bottom, right, left;
} Padding;

typedef struct Layer {
    Address addr;
    struct {
        int height;
        int width;
        int channel;
        int channel_size;
    } input, output;
    struct {
        int output_channel;
        int input_channel;
        int input_channel_size;
    } weights;
    Padding padding;
} Layer;

static Layer tile_create(TileIndex index, TileIndex index_end, Layer body, Layer border, Layer layer, Address addr) {
#define SET_DIM(feature_set, dim, index_name) \
    .dim = index.index_name + 1 == index_end.index_name ? border.feature_set.dim : body.feature_set.dim

    return (Layer) {
        .addr = addr,
        .input = {
            SET_DIM(input, height, height),
            SET_DIM(input, width, width),
            SET_DIM(input, channel, input_channel),
            SET_DIM(input, channel_size, input_channel)
        },
        .output = {
            SET_DIM(output, height, height),
            SET_DIM(output, width, width),
            SET_DIM(output, channel, output_channel),
            SET_DIM(output, channel_size, output_channel)
        },
        .weights = {
            SET_DIM(weights, output_channel, output_channel),
            SET_DIM(weights, input_channel, input_channel),
            SET_DIM(weights, input_channel_size, input_channel)
        },
        .padding = {
            .top = index.height == 0 ? layer.padding.top : 0,
            .bottom = index.height + 1 == index_end.height ? layer.padding.bottom : 0,
            .right = index.width + 1 == index_end.width ? layer.padding.right : 0,
            .left = index.width == 0 ? layer.padding.left : 0
        }
    };
}

#endif // __LAYER_H__
