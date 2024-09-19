# 
# delete.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# SPDX-License-Identifier: Apache-2.0
# 

from manager.platform import PlatformManager
from manager.logbook  import Logbook


def delete(args):

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    logbook = Logbook(args.problem, args.topology)

    logbook.boot_logs_manager(exp_id=args.exp_id)
    logbook.logs_manager.destroy_exp_folder()
