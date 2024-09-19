#!/bin/bash

MYHOME="$(dirname $(dirname "$(readlink -f "${BASH_SOURCE[0]}")"))"

export CC=/usr/bin/gcc #-11
export CXX=/usr/bin/g++ #-11
export GAP_RISCV_GCC_TOOLCHAIN=$MYHOME/gap_riscv_toolchain
export GAP_SDK_HOME=$MYHOME/gap_sdk_private

source $GAP_SDK_HOME/configs/gap9_evk_audio.sh

cmake -B build

cmake --build build -t menuconfig -j 10
cmake --build build -t run CORE=8 platform=gvsoc # --verbose

