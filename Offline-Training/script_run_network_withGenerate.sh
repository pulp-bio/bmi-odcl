#!/bin/bash

MYHOME="$(dirname $(dirname "$(readlink -f "${BASH_SOURCE[0]}")"))"

export CC=/usr/bin/gcc #-11
export CXX=/usr/bin/g++ #-11
export GAP_RISCV_GCC_TOOLCHAIN=$MYHOME/gap_riscv_toolchain
source $MYHOME/gap_sdk_private/configs/gap9_evk_audio.sh #gap9_v2.sh

cd ./dory

python3 network_generate.py Quantlab PULP.GAP9 $MYHOME/Offline-Training/quantlab-cl/examples/fx_integerization/export_to_folder_name/config_MIBMINet.json --optional mixed-sw --app_dir $MYHOME/Offline-Training/export_to_folder_name

source $MYHOME/gap_sdk_private/configs/gap9_evk_audio.sh #gap9_v2.sh

cd $MYHOME/Offline-Training/export_to_folder_name
make run platform=gvsoc
