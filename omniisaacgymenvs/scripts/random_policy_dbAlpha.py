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

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from omniisaacgymenvs.ES.ES_classes import OpenES
from omniisaacgymenvs.ES.feedforward_neural_net import FeedForwardNet

# read Config file
@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # open config files for reading prams
    with open('cfg/ES_config.yml', 'r') as file:
        configs = yaml.safe_load(file)

    # ES parameters configuration
    POPSIZE             = configs['ES_params']['POPSIZE']
    EPISODE_LENGTH      = configs['ES_params']['EPISODE_LENGTH']
    REWARD_FUNCTION     = configs['ES_params']['REWARD_FUNC']
    RANK_FITNESS        = configs['ES_params']['rank_fitness']
    ANTITHETIC          = configs['ES_params']['antithetic']
    LEARNING_RATE       = configs['ES_params']['learning_rate']
    LEARNING_RATE_DECAY = configs['ES_params']['learning_rate_decay']
    SIGMA_INIT          = configs['ES_params']['sigma_init']
    SIGMA_DECAY         = configs['ES_params']['sigma_decay']
    LEARNING_RATE_LIMIT = configs['ES_params']['learning_rate_limit']
    SIGMA_LIMIT         = configs['ES_params']['sigma_limit']

    # Model
    ARCHITECTURE_NAME = configs['Model']['HEBB']['ARCHITECTURE']['name']
    # ARCHITECTURE = configs['Model']['HEBB']['ARCHITECTURE']['size']
    ARCHITECTURE = [4, 16, 8, 1] # for cartpole Env Test

    # Training parameters
    # EPOCHS = configs['Train_params']['EPOCH']
    EPOCHS = 200
    SAVE_EVERY = 100
    TEST = True

    # print("POPSIZE: ", POPSIZE)
    # print("EPISODE_LENGTH: ", EPISODE_LENGTH)
    # print("REWARD_FUNCTION: ", REWARD_FUNCTION)
    # print("ARCHITECTURE_NAME: ", ARCHITECTURE_NAME)
    # print("ARCHITECTURE_size: ", ARCHITECTURE)

    init_net = FeedForwardNet(ARCHITECTURE)
    init_params = init_net.get_params()
    
    print('trainable parameters: ', len(init_params))

    # with open('log_'+str(run)+'.txt', 'a') as outfile:
    #     outfile.write('trainable parameters: ' + str(len(init_params)))

    solver = OpenES(len(init_params),
                    popsize=POPSIZE,
                    rank_fitness=RANK_FITNESS,
                    antithetic=ANTITHETIC,
                    learning_rate=LEARNING_RATE,
                    learning_rate_decay=LEARNING_RATE_DECAY,
                    sigma_init=SIGMA_INIT,
                    sigma_decay=SIGMA_DECAY,
                    learning_rate_limit=LEARNING_RATE_LIMIT,
                    sigma_limit=SIGMA_LIMIT)
    solver.set_mu(init_params)

    # simulation GUI config
    headless = cfg.headless
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # initiate Environment, IsaacGym Simulation
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    # initiate Task, Robot
    task = initialize_task(cfg_dict, env)

    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)
    obs = env.reset()
    # obs_cpu = obs['obs'].cpu().numpy()
    # print("Observation: ", obs)
    actions = torch.zeros(cfg.num_envs)
    # for i in range(cfg.num_envs):
    #     actions[i] = models[i].forward(obs_cpu[i])
    actions = torch.unsqueeze(actions, 1)
    
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)

    if TEST == True:
        for i, file_name in enumerate(res):
            dir_path = './data/model/'
            res = listdir(dir_path)
            trained_data = pickle.load(open('./data/model/'+file_name, 'rb'))
            open_es_data = trained_data[0]
            init_params = open_es_data.mu
            reward = worker_fn(init_params)

            print("reward: ", reward)
    else:
        for epoch in range(EPOCHS):
            run = 'd'

            solutions = solver.ask()

            models = [None] * cfg.num_envs
            for i in range(cfg.num_envs):
                models[i] = FeedForwardNet(ARCHITECTURE)
                models[i].set_params(solutions[i])

            obs_cpu = obs['obs'].cpu().numpy()
            
            for i in range(cfg.num_envs):
                actions[i] = models[i].forward(obs_cpu[i])
            
            for _ in range(50):
                # print('step: ', _)

                obs_cpu = obs['obs'].cpu().numpy()
                            
                for i in range(cfg.num_envs):
                    actions[i] = models[i].forward(obs_cpu[i])
                
                # print("Observation: ", obs)
                # print("Action: ", actions)
                # print("Action_Tensor: ", actions)
                # actions = torch.rand((cfg.num_envs,)+env.action_space.shape, device="cuda:0")
                obs, reward, done, info = env.step(
                    actions # torch.rand((1,)+env.action_space.shape, device="cuda:0")
                )

                # if env._world.is_playing():
                #     if env._world.current_time_step_index == 0:
                #         env._world.reset(soft=True)
                #     actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
                #     env._task.pre_physics_step(actions)
                #     env._world.step(render=render)
                #     env.sim_frame_count += 1
                #     env._task.post_physics_step()
                # else:
                #     env._world.step(render=render)

            # print("reward is", reward)

            reward_cpu = reward.cpu().numpy()
            fitlist = list(reward_cpu)
            solver.tell(fitlist)

            fit_arr = np.array(fitlist)

            print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
            # with open('log_'+str(run)+'.txt', 'a') as outfile:
            #     outfile.write('epoch: ' + str(epoch)
            #             + ' mean: ' + str(fit_arr.mean())
            #             + ' best: ' + str(fit_arr.max())
            #             + ' worst: ' + str(fit_arr.min())
            #             + ' std.: ' + str(fit_arr.std()) + '\n')
                
            pop_mean_curve[epoch] = fit_arr.mean()
            best_sol_curve[epoch] = fit_arr.max()

            if (epoch + 1) % SAVE_EVERY == 0:
                print('saving..')
                pickle.dump((
                    solver,
                    copy.deepcopy(init_net),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open('data/model/FF'+str(run)+'_' + str(len(init_params)) + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))
            
        env._simulation_app.close()




# -------------------------------------------------------------------------------
# --------- Original code with random policy --------------------------------------
# -------------------------------------------------------------------------------
    # # simulation GUI config
    # headless = cfg.headless
    # render = not headless
    # enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # # initiate Environment, IsaacGym Simulation
    # env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    # # sets seed. if seed is -1 will pick a random one
    # from omni.isaac.core.utils.torch.maths import set_seed
    # cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    # cfg_dict['seed'] = cfg.seed
    # # initiate Task, Robot
    # task = initialize_task(cfg_dict, env)

    # # Simulation Loop
    # while env._simulation_app.is_running():
    #     if env._world.is_playing():
    #         if env._world.current_time_step_index == 0:
    #             env._world.reset(soft=True)
    #         actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #         env._task.pre_physics_step(actions)
    #         env._world.step(render=render)
    #         env.sim_frame_count += 1
    #         env._task.post_physics_step()
    #     else:
    #         env._world.step(render=render)

    # env._simulation_app.close()
# -------------------------------------------------------------------------------

if __name__ == '__main__':
    parse_hydra_configs()
