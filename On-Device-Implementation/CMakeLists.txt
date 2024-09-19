# CMakeLists.txt
# Alessio Burrello <alessio.burrello@unibo.it>
# Cristian Cioflan <cioflanc@iis.ee.ethz.ch>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
# Lan Mei <lanmei@student.ethz.ch>
#
# Copyright (C) 2019-2024 ETH Zurich and University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.19.5)

set(TARGET_NAME         "main")
set(MAIN                "main.c") # main_LwF.c for LwF

include ($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
set(CONFIG_GAP_SDK_HOME $ENV{GAP_SDK_HOME})

set(BOARD_NAME gap9_evk)
set(PMSIS_OS freertos)
set(platform gvsoc)
set (FLASH_TYPE DEFAULT)
set (RAM_TYPE   DEFAULT)
set (NUM_CORES 8)

set(DORY_NET_DIR        ${CMAKE_CURRENT_LIST_DIR}/Backbone-Example/) 
set(CURDIR              ${CMAKE_CURRENT_LIST_DIR})
set(BUILD_DIR           ${CURDIR}/build)
set(TRAINLIB_DIR        ${CMAKE_CURRENT_LIST_DIR}/pulp-trainlib)
set(TRAINLIB_FILES_DIR  ${CURDIR}/Classifier-Example) # Classifier-Example-LwF for LwF

# PULP trainlib
set(MATMUL_TYPE_FW_L0  0)         # Selects which optimized matmul to be used in FW (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)
set(MATMUL_TYPE_WG_L0  0)         # Selects which optimized matmul to be used in WEIGHT GRAD (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)
set(MATMUL_TYPE_IG_L0  0)         # Selects which optimized matmul to be used in IN GRAD (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)

set (NUM_MATMULS 24)
set (MATMUL_TYPE 0)
set (NUM_SIZES 3)

set (IN_CH 928)
set (OUT_CH 2)
set (STEP FORWARD_BACKWARD_PROP)

list (APPEND TARGET_SRCS 
                         ${MAIN}
)
list (APPEND TARGET_INCS 
                         -I${DORY_NET_DIR}inc
                         -I$ENV{GAP_SDK_HOME}rtos/pmsis/bsp/include
                         -I$ENV{GAP_SDK_HOME}rtos/pmsis/bsp/include/bsp
                         -I$ENV{GAP_SDK_HOME}rtos/pmsis/bsp/include/bsp/ram
                         -I$ENV{GAP_SDK_HOME}rtos/pmsis/bsp/include/bsp/flash
                         -I$ENV{GAP_SDK_HOME}rtos/pmsis/bsp/include/bsp/flash/hyperflash
                         -I$ENV{GAP_SDK_HOME}rtos/pmsis/bsp/include/bsp/fs
                         -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                         -I${TRAINLIB_DIR}/lib/include
                         -I${TRAINLIB_FILES_DIR}/

)

list (APPEND TARGET_SRCS 
                         
                         ${DORY_NET_DIR}src/BNReluConvolution0.c
                         ${DORY_NET_DIR}src/BNReluConvolution1.c 
                         ${DORY_NET_DIR}src/BNReluConvolution3.c 
                         ${DORY_NET_DIR}src/BNReluConvolution4.c 
                         ${DORY_NET_DIR}src/dory_dma.c
                         ${DORY_NET_DIR}src/FullyConnected6.c 
                         ${DORY_NET_DIR}src/mem.c 
                         ${DORY_NET_DIR}src/net_utils.c 
                         ${DORY_NET_DIR}src/network.c 
                         ${DORY_NET_DIR}src/Pooling2.c 
                         ${DORY_NET_DIR}src/Pooling5.c 
                         ${DORY_NET_DIR}src/pulp_nn_avgpool_u8_u8.c 
                         ${DORY_NET_DIR}src/pulp_nn_conv_i8_i8_i8.c 
                         ${DORY_NET_DIR}src/pulp_nn_conv_i8_u8_i8.c 
                         ${DORY_NET_DIR}src/pulp_nn_depthwise_i8_u8_i8.c
                         ${DORY_NET_DIR}src/pulp_nn_depthwise_u8_i8_i8.c
                         ${DORY_NET_DIR}src/pulp_nn_linear_u8_i32_i8.c 
                         ${DORY_NET_DIR}src/pulp_nn_matmul_i8_i8_i8.c 
                         ${DORY_NET_DIR}src/pulp_nn_matmul_i8_u8_i8.c 
                         ${DORY_NET_DIR}src/pulp_nn_matmul_u8_i8_i8.c              
)

list (APPEND TARGET_SRCS ${TRAINLIB_FILES_DIR}/net.c
                         ${TRAINLIB_FILES_DIR}/pulp_losses_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_act_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_act_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_conv2d_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_conv2d_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_conv_dw_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_conv_dw_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_conv_pw_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_conv_pw_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_im2col_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_im2col_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_linear_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_linear_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_losses_fp16.c
                         # ${TRAINLIB_DIR}/lib/sources/pulp_losses_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_matmul_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_matmul_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_mhsa_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_optimizers_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_optimizers_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_pooling_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_pooling_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_rnn_fp32.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_train_utils_fp16.c
                         ${TRAINLIB_DIR}/lib/sources/pulp_train_utils_fp32.c
)



list(APPEND TARGET_CFLAGS   -O2 #-fno-indirect-inlining
                            -s
                            #-w
                            -mno-memcpy
                            -fno-tree-loop-distribute-patterns
)


list(APPEND TARGET_PREPROCESSOR 
                		-DTARGET_CHIP_FAMILY_GAP9
                	        -DGAP_SDK=1
                                -DNUM_CORES=${NUM_CORES}
                                
                                -DFLASH_TYPE=${FLASH_TYPE}
                                -DUSE_${FLASH_TYPE}
                                -DUSE_${RAM_TYPE}
                                
				-DCONFIG_GAP9_EVK
                                
                                -DFS_READ_FS
                                #-DCONFIG_FLASH_LAYOUT_CUSTOM
                                -DSINGLE_CORE_DMA
                                
                                -DOPTIMIZE
                                -DMATMUL_TYPE=${MATMUL_TYPE}
                                
                                -DSTATS
                                #-DPROF_NET
                                -DMEMOCC_COMP
                                -DVERBOSE
                                
)

list(APPEND READFS_FILES 
                        ${DORY_NET_DIR}hex/BNReluConvolution0_weights.hex
                        ${DORY_NET_DIR}hex/BNReluConvolution1_weights.hex
                        ${DORY_NET_DIR}hex/BNReluConvolution3_weights.hex
                        ${DORY_NET_DIR}hex/BNReluConvolution4_weights.hex
                        ${DORY_NET_DIR}hex/FullyConnected6_weights.hex
                        ${DORY_NET_DIR}hex/inputs_0.hex
                        ${DORY_NET_DIR}hex/inputs_1.hex
                        ${DORY_NET_DIR}hex/inputs_2.hex
                        ${DORY_NET_DIR}hex/inputs_3.hex
                        ${DORY_NET_DIR}hex/inputs_4.hex
                        ${DORY_NET_DIR}hex/inputs_5.hex
                        ${DORY_NET_DIR}hex/inputs_6.hex
                        ${DORY_NET_DIR}hex/inputs_7.hex
                        ${DORY_NET_DIR}hex/inputs_8.hex
                        ${DORY_NET_DIR}hex/inputs_9.hex
                        ${DORY_NET_DIR}hex/inputs_10.hex
                        ${DORY_NET_DIR}hex/inputs_11.hex
                        ${DORY_NET_DIR}hex/inputs_12.hex
                        ${DORY_NET_DIR}hex/inputs_13.hex
                        ${DORY_NET_DIR}hex/inputs_14.hex
                        ${DORY_NET_DIR}hex/inputs_15.hex
                        ${DORY_NET_DIR}hex/inputs_16.hex
                        ${DORY_NET_DIR}hex/inputs_17.hex
                        ${DORY_NET_DIR}hex/inputs_18.hex
                        ${DORY_NET_DIR}hex/inputs_19.hex
                        ${DORY_NET_DIR}hex/inputs_20.hex
                        ${DORY_NET_DIR}hex/inputs_21.hex
                        ${DORY_NET_DIR}hex/inputs_22.hex
                        ${DORY_NET_DIR}hex/inputs_23.hex
                        ${DORY_NET_DIR}hex/inputs_24.hex
                        ${DORY_NET_DIR}hex/inputs_25.hex
                        ${DORY_NET_DIR}hex/inputs_26.hex
                        ${DORY_NET_DIR}hex/inputs_27.hex
                        ${DORY_NET_DIR}hex/inputs_28.hex
                        ${DORY_NET_DIR}hex/inputs_29.hex
                        ${DORY_NET_DIR}hex/inputs_30.hex
                        ${DORY_NET_DIR}hex/inputs_31.hex
                        ${DORY_NET_DIR}hex/inputs_32.hex
                        ${DORY_NET_DIR}hex/inputs_33.hex
                        ${DORY_NET_DIR}hex/inputs_34.hex
                        ${DORY_NET_DIR}hex/inputs_35.hex
                        ${DORY_NET_DIR}hex/inputs_36.hex
                        ${DORY_NET_DIR}hex/inputs_37.hex
                        ${DORY_NET_DIR}hex/inputs_38.hex
                        ${DORY_NET_DIR}hex/inputs_39.hex
                        ${DORY_NET_DIR}hex/inputs_40.hex
                        ${DORY_NET_DIR}hex/inputs_41.hex
                        ${DORY_NET_DIR}hex/inputs_42.hex
                        ${DORY_NET_DIR}hex/inputs_43.hex
                        ${DORY_NET_DIR}hex/inputs_44.hex
                        ${DORY_NET_DIR}hex/inputs_45.hex
                        ${DORY_NET_DIR}hex/inputs_46.hex
                        ${DORY_NET_DIR}hex/inputs_47.hex
                        ${DORY_NET_DIR}hex/inputs_48.hex
                        ${DORY_NET_DIR}hex/inputs_49.hex
                        ${DORY_NET_DIR}hex/inputs_50.hex
                        ${DORY_NET_DIR}hex/inputs_51.hex
                        ${DORY_NET_DIR}hex/inputs_52.hex
                        ${DORY_NET_DIR}hex/inputs_53.hex
                        ${DORY_NET_DIR}hex/inputs_54.hex
                        ${DORY_NET_DIR}hex/inputs_55.hex
                        ${DORY_NET_DIR}hex/inputs_56.hex
                        ${DORY_NET_DIR}hex/inputs_57.hex
                        ${DORY_NET_DIR}hex/inputs_58.hex
                        ${DORY_NET_DIR}hex/inputs_59.hex
                        ${DORY_NET_DIR}hex/inputs_60.hex
                        ${DORY_NET_DIR}hex/inputs_61.hex
                        ${DORY_NET_DIR}hex/inputs_62.hex
                        ${DORY_NET_DIR}hex/inputs_63.hex
                        ${DORY_NET_DIR}hex/inputs_64.hex
                        ${DORY_NET_DIR}hex/inputs_65.hex
                        ${DORY_NET_DIR}hex/inputs_66.hex
                        ${DORY_NET_DIR}hex/inputs_67.hex
                        ${DORY_NET_DIR}hex/inputs_68.hex
                        ${DORY_NET_DIR}hex/inputs_69.hex
                        ${DORY_NET_DIR}hex/inputs_70.hex
                        ${DORY_NET_DIR}hex/inputs_71.hex
                        ${DORY_NET_DIR}hex/inputs_72.hex
                        ${DORY_NET_DIR}hex/inputs_73.hex
                        ${DORY_NET_DIR}hex/inputs_74.hex
                        ${DORY_NET_DIR}hex/inputs_75.hex
                        ${DORY_NET_DIR}hex/inputs_76.hex
                        ${DORY_NET_DIR}hex/inputs_77.hex
                        ${DORY_NET_DIR}hex/inputs_78.hex
                        ${DORY_NET_DIR}hex/inputs_79.hex
                        ${DORY_NET_DIR}hex/inputs_80.hex
                        ${DORY_NET_DIR}hex/inputs_81.hex
                        ${DORY_NET_DIR}hex/inputs_82.hex
                        ${DORY_NET_DIR}hex/inputs_83.hex
                        ${DORY_NET_DIR}hex/inputs_84.hex
                        ${DORY_NET_DIR}hex/inputs_85.hex
                        ${DORY_NET_DIR}hex/inputs_86.hex
                        ${DORY_NET_DIR}hex/inputs_87.hex
                        ${DORY_NET_DIR}hex/inputs_88.hex
                        ${DORY_NET_DIR}hex/inputs_89.hex
                        ${DORY_NET_DIR}hex/inputs_90.hex
                        ${DORY_NET_DIR}hex/inputs_91.hex
                        ${DORY_NET_DIR}hex/inputs_92.hex
                        ${DORY_NET_DIR}hex/inputs_93.hex
                        ${DORY_NET_DIR}hex/inputs_94.hex
                        ${DORY_NET_DIR}hex/inputs_95.hex
                        ${DORY_NET_DIR}hex/inputs_96.hex
                        ${DORY_NET_DIR}hex/inputs_97.hex
                        ${DORY_NET_DIR}hex/inputs_98.hex
                        ${DORY_NET_DIR}hex/inputs_99.hex
)


project (${TARGET_NAME} C ASM)

add_executable (${TARGET_NAME} ${TARGET_SRCS})

target_compile_options(${TARGET_NAME} PUBLIC   ${TARGET_CFLAGS}
                                               ${TARGET_INCS}
                                               ${TARGET_PREPROCESSOR}

)

target_link_libraries(${TARGET_NAME} PRIVATE m)

setupos(${TARGET_NAME})

