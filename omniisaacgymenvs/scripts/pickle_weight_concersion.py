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
import wandb
import time
import timeit

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from omniisaacgymenvs.ES.ES_classes import OpenES, CMAES
from omniisaacgymenvs.ES.feedforward_neural_net_gpu import FeedForwardNet
from omniisaacgymenvs.ES.hebbian_neural_net import HebbianNet
from omniisaacgymenvs.ES.rbf_neural_net import RBFNet
from omniisaacgymenvs.ES.rbf_hebbian_neural_net import RBFHebbianNet

# read Config file
@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # open config files for reading prams
    #with open('cfg/ES_config.yml', 'r') as file:
    #    configs = yaml.safe_load(file)

    # ES parameters configuration
    POPSIZE             = cfg.num_envs # configs['ES_params']['POPSIZE']
    # EPISODE_LENGTH      = cfg.ES_params.EPISODE_LENGTH
    RANK_FITNESS        = cfg.ES_params.rank_fitness
    ANTITHETIC          = cfg.ES_params.antithetic
    LEARNING_RATE       = cfg.ES_params.learning_rate
    LEARNING_RATE_DECAY = cfg.ES_params.learning_rate_decay
    SIGMA_INIT          = cfg.ES_params.sigma_init
    SIGMA_DECAY         = cfg.ES_params.sigma_decay
    LEARNING_RATE_LIMIT = cfg.ES_params.learning_rate_limit
    SIGMA_LIMIT         = cfg.ES_params.sigma_limit

    # Model
    ARCHITECTURE_NAME = cfg.model
    ARCHITECTURE_TYPE = cfg.RBFHebb_model_type
    # ARCHITECTURE = configs['Model']['HEBB']['ARCHITECTURE']['size']
    ARCHITECTURE = cfg.ARCHITECTURE
    RBF_ARCHITECTURE = cfg.RBF_ARCHITECTURE

    # Training parameters
    # EPOCHS = configs['Train_params']['EPOCH']
    EPOCHS = cfg.EPOCHS
    EPISODE_LENGTH = cfg.EPISODE_LENGTH
    SAVE_EVERY = cfg.SAVE_EVERY
    USE_TRAIN_PARAMS = cfg.USE_TRAIN_PARAMS
    wandb_activate = cfg.wandb_activate
    TASK = cfg["task_name"]
    experiment = cfg.experiment
    test_env = cfg.test_env
    rew = cfg.rewards_type

    exp_name = cfg.model+'_'+TASK+experiment+'_Exp_'+str(1)+rew
    if wandb_activate:
        # wandb.init(project='Cartpole_ES_log',
        # wandb.init(project='Ant_ES_log',
        wandb.init(project='dbAlpha_ES_New_log',
                    name=exp_name, 
                    config=cfg_dict)
    
    # print("POPSIZE: ", POPSIZE)
    # print("EPISODE_LENGTH: ", EPISODE_LENGTH)
    # print("REWARD_FUNCTION: ", REWARD_FUNCTION)
    # print("ARCHITECTURE_NAME: ", ARCHITECTURE_NAME)
    # print("ARCHITECTURE_size: ", ARCHITECTURE)

    # if ARCHITECTURE_NAME == 'Feedforward':
    #     models = FeedForwardNet(ARCHITECTURE, POPSIZE)
    #     # init_net = FeedForwardNet(ARCHITECTURE, POPSIZE)
    # elif ARCHITECTURE_NAME == 'Hebb':
    #     models = HebbianNet(ARCHITECTURE, POPSIZE)
    # elif ARCHITECTURE_NAME == 'rbf':
    #     models = RBFNet(POPSIZE, RBF_ARCHITECTURE[1], RBF_ARCHITECTURE[0], 'obj_trans')
    # elif ARCHITECTURE_NAME == 'Hebb_rbf':
    #     models = RBFHebbianNet(POPSIZE, RBF_ARCHITECTURE[1], RBF_ARCHITECTURE[0], ARCHITECTURE_TYPE)
    # init_params = models.get_params_a_model()


    # with open('log_'+str(run)+'.txt', 'a') as outfile:
    #     outfile.write('trainable parameters: ' + str(len(init_params)))

    # solver = OpenES(len(init_params),
    #                 popsize=POPSIZE,
    #                 rank_fitness=RANK_FITNESS,
    #                 antithetic=ANTITHETIC,
    #                 learning_rate=LEARNING_RATE,
    #                 learning_rate_decay=LEARNING_RATE_DECAY,
    #                 sigma_init=SIGMA_INIT,
    #                 sigma_decay=SIGMA_DECAY,
    #                 learning_rate_limit=LEARNING_RATE_LIMIT,
    #                 sigma_limit=SIGMA_LIMIT)
    # solver.set_mu(init_params)

    # print(np.array([POPSIZE]).ndim)
    # solver = CMAES(init_params,  # number of model parameters
    #                sigma_init=0.4,  # initial standard deviation
    #                popsize=POPSIZE,  # population size
    #                weight_decay=0.995)  # weight decay coefficient
    # solver.set_mu(init_params)

    print('TASK', TASK)
    print('exp_name', exp_name)
    print('experiment', experiment)
    print('test_env', test_env)
    print('model: ', ARCHITECTURE_NAME)
    print('model size: ', ARCHITECTURE)
    # print('trainable parameters: ', len(init_params))
    # print("Action space is", env.action_space)
    # obs = env.reset()

    # obs_cpu = obs['obs'].cpu().numpy()
    # print("Observation: ", obs)
    # actions = torch.ones(cfg.num_envs, ARCHITECTURE[-1])
    # total_rewards = torch.zeros(cfg.num_envs)
    # total_rewards = torch.unsqueeze(total_rewards, 0)
    # total_rewards = total_rewards.cuda()
    # print("Action: ", actions)
    # for i in range(cfg.num_envs):
    #     actions[i] = models[i].forward(obs_cpu[i])
    # actions = torch.unsqueeze(actions, 1)
    # print("Action2: ", actions)
    # actions = actions.cuda()
    # print("Action2_cuda: ", actions)

    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)


    if ARCHITECTURE_NAME == 'Feedforward':
        dir_path = './data/'+TASK+'/model/FF/'
    elif ARCHITECTURE_NAME == 'Hebb':
        dir_path = './data/'+TASK+'/model/test/good_hebb/'
    elif ARCHITECTURE_NAME == 'rbf':
        dir_path = 'data/'+TASK+'/model/rbf/'
    elif ARCHITECTURE_NAME == 'Hebb_rbf':
        dir_path = 'data/'+TASK+'/model/Hebb_rbf/' # test_hebb_params/'


    res = listdir(dir_path)
    for i, file_name in enumerate(sorted(res[0:1])):
        file_name = 'Feedforward_dbAlpha6legs_walk_Exp_1vx_d_4352499_300.42620849609375.pickle'
        # file_name = 'rbf_dbAlphadbAlpha_newGaitRew_Exp_1newGaitRew_d_180497_260.62725830078125.pickle'
        # file_name = 'Hebb_dbAlpha_objectsmallbox_trans_Exp_1-vx_d_18240249_306.11181640625.pickle'
        print('file_name: ', file_name)

        # Load Data script
        # time.sleep(2)
        trained_data = pickle.load(open(dir_path+file_name, 'rb'))
        open_es_data = trained_data[0]
        # init_params = open_es_data.best_mu # best_mu   
        init_params = open_es_data.best_param() # best_mu   
            
        # print('trained_data: ', trained_data)
        # print('init_params: ', init_params)
        # models.set_params_single_model(init_params)            
        # models = [None] * cfg.num_envs
        # for i in range(cfg.num_envs):
        #     models[i] = FeedForwardNet(ARCHITECTURE)
        #     models[i].set_params(solutions[i])
        # solutions = open_es_data.ask()
        np.save('np_array/'+ARCHITECTURE_NAME+'_'+experiment+'_Best_weight', init_params)

if __name__ == '__main__':
    parse_hydra_configs()
