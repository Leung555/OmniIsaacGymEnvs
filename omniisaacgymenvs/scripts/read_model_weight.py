# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import pickle
import torch
import hydra
from omegaconf import DictConfig
import yaml
import copy
from os import listdir
import timeit

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task

# read Config file
@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # open config files for reading prams
    with open('cfg/ES_config.yml', 'r') as file:
        configs = yaml.safe_load(file)

    # Model
    ARCHITECTURE_NAME = cfg.model
    # ARCHITECTURE = configs['Model']['HEBB']['ARCHITECTURE']['size']
    # ARCHITECTURE = [102, 256, 128, 18] # for cartpole Env Test
    ARCHITECTURE = [4, 16, 8, 1] # for cartpole Env Test

    TASK = cfg["task_name"]
    print('TASK', TASK)
    if ARCHITECTURE_NAME == 'Feedforward':
        dir_path = './data/'+TASK+'/model/FF/'
    elif ARCHITECTURE_NAME == 'Hebb':
        dir_path = './data/'+TASK+'/model/Hebb/'
    res = listdir(dir_path)
    res.sort()
    TEST = cfg.test
    for i, file_name in enumerate(res):
        print('file_name: ', file_name)

    print("Last File: ", res[-1])

if __name__ == '__main__':
    parse_hydra_configs()
