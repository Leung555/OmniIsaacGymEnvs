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

    if ARCHITECTURE_NAME == 'Feedforward':
        models = FeedForwardNet(ARCHITECTURE, POPSIZE)
        # init_net = FeedForwardNet(ARCHITECTURE, POPSIZE)
    elif ARCHITECTURE_NAME == 'Hebb':
        models = HebbianNet(ARCHITECTURE, POPSIZE)
    elif ARCHITECTURE_NAME == 'rbf':
        models = RBFNet(POPSIZE, RBF_ARCHITECTURE[1], RBF_ARCHITECTURE[0])
    elif ARCHITECTURE_NAME == 'Hebb_rbf':
        models = RBFHebbianNet(POPSIZE, RBF_ARCHITECTURE[1], RBF_ARCHITECTURE[0], ARCHITECTURE_TYPE)
    init_params = models.get_params_a_model()


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

    # print(np.array([POPSIZE]).ndim)
    # solver = CMAES(init_params,  # number of model parameters
    #                sigma_init=0.4,  # initial standard deviation
    #                popsize=POPSIZE,  # population size
    #                weight_decay=0.995)  # weight decay coefficient
    # solver.set_mu(init_params)

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

    print('TASK', TASK)
    print('exp_name', exp_name)
    print('model: ', ARCHITECTURE_NAME)
    print('model size: ', ARCHITECTURE)
    print('trainable parameters: ', len(init_params))
    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)
    # print("Action space is", env.action_space)
    # obs = env.reset()

    # obs_cpu = obs['obs'].cpu().numpy()
    # print("Observation: ", obs)
    actions = torch.ones(cfg.num_envs, ARCHITECTURE[-1])
    total_rewards = torch.zeros(cfg.num_envs)
    # total_rewards = torch.unsqueeze(total_rewards, 0)
    total_rewards = total_rewards.cuda()
    # print("Action: ", actions)
    # for i in range(cfg.num_envs):
    #     actions[i] = models[i].forward(obs_cpu[i])
    # actions = torch.unsqueeze(actions, 1)
    # print("Action2: ", actions)
    actions = actions.cuda()
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
    # dir_path = listdir(dir_path+'simpleRBFHebb'+'/')
    if USE_TRAIN_PARAMS:
        res = listdir(dir_path)
        for i, file_name in enumerate(res[0:1]):
            file_name = 'rbf_dbAlpha6legs_Hebb_lowpassAction_uneven_Exp_1ant_setup_d_180499_106.85049438476562.pickle'
            print('file_name: ', file_name)
            trained_data = pickle.load(open(dir_path+file_name, 'rb'))
            open_es_data = trained_data[0]
            init_params = open_es_data.best_param() # best_mu   
            solver = open_es_data
            solver.set_mu(init_params)

    TEST = cfg.test
    if TEST == True:
        res = listdir(dir_path)
        for i, file_name in enumerate(sorted(res)):
            # file_name = 'rbf_dbAlphadbAlpha_newGaitRew_Exp_1newGaitRew_d_180497_260.62725830078125.pickle'
            # file_name = 'Feedforward_dbAlpha6legs_Hebb_Wmaxnormalize_Exp_1ant_setup_d_4352499_260.0474853515625.pickle'
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
            obs = env.reset()
            
            models.set_params_single_model(init_params)

            total_rewards = torch.zeros(cfg.num_envs)
            total_rewards = total_rewards.cuda()

            obs = env.reset()
            w1 = []
            w2 = []
            w3 = []
            act = []

            for _ in range(EPISODE_LENGTH):
                # print('step: ', _)
                ############### CPU Version ###############
                # TODO
                actions = models.forward(obs['obs'])
                # print('actions: ', actions)
                ###########################################
                # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
                # print("Action_3: ", actions)
                obs, reward, done, info = env.step(
                    actions
                )
                ###########################################
                ############### CPU Version Multiple models ###############
                # obs_cpu = obs['obs'].cpu().numpy()
                # for i in range(cfg.num_envs):
                #     actions[i] = models[i].forward(obs_cpu[i])
                ##########################################################

                ############### GPU Version ###############
                # for i in range(cfg.num_envs):
                #     actions[i] = init_net.forward(obs['obs'][i])
                ###########################################
                # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)                # print("Action3: ", actions)
                
                # Weight collection ######
                # weight = models.get_weights()
                # w1.append(weight[0].cpu().numpy())
                # w2.append(weight[1].cpu().numpy())
                # w3.append(weight[2].cpu().numpy())
                # w3.append(weight[2].cpu().numpy())
                # act.append(actions.cpu().numpy())

                total_rewards += reward

            # print("reward is", reward)
            print('total_rewards: ', total_rewards)
            # data_np = np.array(act)

            # Weight save ######
            # w1 = np.array(w1)
            # w2 = np.array(w2)
            # w3 = np.array(w3)
            # # np.save('np_array/actions_ff', data_np)
            # # np.save('np_array/w1_ff', w1)
            # # np.save('np_array/w2_ff', w2)
            # # np.save('np_array/w3_ff', w3)

            # np.save('np_array/actions_hebb', data_np)
            # np.save('np_array/w1_hebb', w1)
            # np.save('np_array/w2_hebb', w2)
            # np.save('np_array/w3_hebb', w3)
            # np.save('np_array/param_hebb', init_params)

    else:
        initial_time = timeit.default_timer()
        print("initial_time", initial_time)
        time_per_epoch_array = []

        for epoch in range(EPOCHS):
            # print('Epoch: ', epoch)
            start_time = timeit.default_timer()
            run = 'd'

            # retrieve solution from ES 
            solutions = solver.ask()
            
            # set models parameters 
            models.set_params(solutions)

            obs = env.reset()
            # obs = obs['obs'].cpu().numpy()
            
            total_rewards = torch.zeros(cfg.num_envs)
            total_rewards = total_rewards.cuda()
            # for i in range(cfg.num_envs):
            #     actions[i] = models[i].forward(obs['obs'][i])
            
            for _ in range(EPISODE_LENGTH):
                # print('step: ', _)
                ############### CPU Version ###############
                # obs_cpu = obs['obs'].cpu().numpy()
                # # print('obs_cpu', obs_cpu)
                # # print("Observation: ", obs)
                # for i in range(cfg.num_envs):
                #     actions[i] = models[i].forward(obs_cpu[i])
                ###########################################
                ############### GPU Version ###############
                # obs = torch.zeros(POPSIZE, ARCHITECTURE[0]).cuda()
                actions = models.forward(obs['obs'])
                # print('actions: ', actions[0, 0:6])
                ###########################################
                # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
                # print("Action_3: ", actions)
                obs, reward, done, info = env.step(
                    actions
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
                # print('reward: ', reward)
                # print('total_rewards: ', total_rewards)
                total_rewards += reward

            # print("reward is", reward)
            # print('total_rewards: ', total_rewards)

            total_rewards_cpu = total_rewards.cpu().numpy()
            fitlist = list(total_rewards_cpu)
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

            # WanDB Log data -------------------------------
            if wandb_activate:
                wandb.log({"epoch": epoch,
                            "mean" : np.mean(fitlist),
                            "best" : np.max(fitlist),
                            "worst": np.min(fitlist),
                            "std"  : np.std(fitlist),
                            })
            # -----------------------------------------------

            if (epoch + 1) % SAVE_EVERY == 0:
                print('saving..')
                pickle.dump((
                    solver,
                    copy.deepcopy(models),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open(dir_path+exp_name+'_'+str(run)+'_' + str(len(init_params)) + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))
            
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
