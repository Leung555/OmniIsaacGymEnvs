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

import argparse
import pdb
#from line_profiler import LineProfiler
import torch.nn as nn
from torch.autograd import Variable
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import platform

import numpy as np

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from omniisaacgymenvs.ES.ES_classes import OpenES, CMAES
from omniisaacgymenvs.ES.feedforward_neural_net_gpu import FeedForwardNet
from omniisaacgymenvs.ES.hebbian_neural_net import HebbianNet
from omniisaacgymenvs.ES.rbf_neural_net import RBFNet
from omniisaacgymenvs.ES.rbf_hebbian_neural_net import RBFHebbianNet



np.set_printoptions(precision=4)
ADDITIONALINPUTS = 4 # 1 input for the previous reward, 1 input for numstep, 1 unused,  1 "Bias" input

NBACTIONS = 1   # U, D, L, R

RFSIZE = 3      # Receptive Field: RFSIZE x RFSIZE

TOTALNBINPUTS =  4

class Network(nn.Module):
    
    def __init__(self, isize, hsize): 
        # NBACTIONS = 1
        super(Network, self).__init__()
        self.hsize, self.isize  = hsize, isize 

        self.i2h = torch.nn.Linear(isize, hsize)    # Weights from input to recurrent layer
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Baseline (non-plastic) component of the plastic recurrent layer
        
        self.alpha =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))   # Plasticity coefficients of the plastic recurrent layer; one alpha coefficient per recurrent connection
        #self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))  # Per-neuron alpha
        #self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))         # Single alpha for whole network

        self.h2mod = torch.nn.Linear(hsize, 1)      # Weights from the recurrent layer to the (single) neurodulator output
        self.modfanout = torch.nn.Linear(1, hsize)  # The modulator output is passed through a different 'weight' for each neuron (it 'fans out' over neurons)

        self.h2o = torch.nn.Linear(hsize, NBACTIONS)    # From recurrent to outputs (action probabilities)
        self.h2v = torch.nn.Linear(hsize, 1)            # From recurrent to value-prediction (used for A2C)


        
    def forward(self, inputs, hidden): # hidden is a tuple containing the h-state (i.e. the recurrent hidden state) and the hebbian trace 
            HS = self.hsize
        
            # hidden[0] is the h-state; hidden[1] is the Hebbian trace
            hebb = hidden[1]


            # Each *column* of w, alpha and hebb contains the inputs weights to a single neuron
            hactiv = torch.tanh( self.i2h(inputs) + hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)  )  # Update the h-state
            activout = self.h2o(hactiv)  # Pure linear, raw scores - to be softmaxed later, outside the function
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), hactiv.unsqueeze(1))  # Batched outer product of previous hidden state with new hidden state
            
            # We also need to compute the eta (the plasticity rate), wich is determined by neuromodulation
            # Note that this is "simple" neuromodulation.
            myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
            
            # The neuromodulated eta is passed through a vector of fanout weights, one per neuron.
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell.
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
            myeta = self.modfanout(myeta) 
            
            
            # Updating Hebbian traces, with a hard clip (other choices are possible)
            self.clipval = 2.0
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)

            hidden = (hactiv, hebb)
            return activout, valueout, hidden




    def initialZeroState(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize), requires_grad=False )

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self, BATCHSIZE):
        return Variable(torch.zeros(BATCHSIZE, self.hsize, self.hsize) , requires_grad=False)



# That's it for plasticity! The rest of the code simply implements the maze task and the A2C RL algorithm.


# read Config file
@hydra.main(config_name="config_bp", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # open config files for reading prams
    #with open('cfg/ES_config.yml', 'r') as file:
    #    configs = yaml.safe_load(file)

    # BP parameters
    rngseed =   cfg.Backpropamine.rngseed 
    rew_bp =    cfg.Backpropamine.rew_bp 
    wp =        cfg.Backpropamine.wp 
    bent =      cfg.Backpropamine.bent 
    blossv =    cfg.Backpropamine.blossv 
    msize =     cfg.Backpropamine.msize 
    gr =        cfg.Backpropamine.gr 
    gc =        cfg.Backpropamine.gc 
    lr =        cfg.Backpropamine.lr 
    eplen =     cfg.Backpropamine.eplen 
    hs =        cfg.Backpropamine.hs 
    bs =        cfg.Backpropamine.bs 
    l2 =        cfg.Backpropamine.l2 
    nbiter =    cfg.Backpropamine.nbiter 
    save_every =cfg.Backpropamine.save_every 
    pe =        cfg.Backpropamine.pe 

    #params = dict(click.get_current_context().params)

    # ADDITIONALINPUTS = 4 # 1 input for the previous reward, 1 input for numstep, 1 unused,  1 "Bias" input
    # NBACTIONS = 4   # U, D, L, R
    # RFSIZE = 3      # Receptive Field: RFSIZE x RFSIZE
    # TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDITIONALINPUTS + NBACTIONS
    
    print("Starting training...")
    params = {}
    #params.update(defaultParams)
    # params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    # suffix = "btchFixmod_"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'pe' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(rngseed)   # Turning the parameters into a nice suffix for filenames
    suffix = "test"

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(rngseed); random.seed(rngseed); torch.manual_seed(rngseed)

    print("Initializing network")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    net = Network(TOTALNBINPUTS, hs).to(device)  # Creating the network
    
    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    #total_loss = 0.0
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*lr, eps=1e-4, weight_decay=l2)
    #optimizer = torch.optim.SGD(net.parameters(), lr=1.0*params['lr'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['steplr'])

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

    # print('TASK', TASK)
    # print('exp_name', exp_name)
    # print('model: ', ARCHITECTURE_NAME)
    # print('model size: ', ARCHITECTURE)
    # print('trainable parameters: ', len(init_params))
    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)


    BATCHSIZE = bs

    # LABSIZE = msize 
    # lab = np.ones((LABSIZE, LABSIZE))
    # CTR = LABSIZE // 2 


    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    # meanrewards = np.zeros((LABSIZE, LABSIZE))
    # meanrewardstmp = np.zeros((LABSIZE, LABSIZE, eplen))


    # pos = 0
    hidden = net.initialZeroState(BATCHSIZE)
    hebb = net.initialZeroHebb(BATCHSIZE)
    print(hidden.shape)
    print(hebb.shape)

    print("Starting episodes!")

    for numiter in range(nbiter):

        PRINTTRACE = 0
        if (numiter+1) % (pe) == 0:
            PRINTTRACE = 1
        
        optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState(BATCHSIZE).to(device)
        hebb = net.initialZeroHebb(BATCHSIZE).to(device)
        # print(hidden.shape)
        # print(hebb.shape)

        # numactionchosen = 0


        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        obs = env.reset()

        #reloctime = np.random.randint(params['eplen'] // 4, (3 * params['eplen']) // 4)

        #print("EPISODE ", numiter)
        for numstep in range(eplen):
            
            # inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32') 
            
            # for nb in range(BATCHSIZE):
            
            #     actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)

            #     # Step Simulation
            #     obs, reward, done, info = env.step(
            #         actions
            #     )
            #     inputs[nb] = actions
        
            #     # inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE//2:posr[nb] + RFSIZE//2 +1, posc[nb] - RFSIZE //2:posc[nb] + RFSIZE//2 +1].flatten() * 1.0
                
            #     # Previous chosen action
                # inputs[nb, RFSIZE * RFSIZE +1] = 1.0 # Bias neuron
            #     inputs[nb, RFSIZE * RFSIZE +2] = numstep / eplen
            #     inputs[nb, RFSIZE * RFSIZE +3] = 1.0 * reward[nb]
            #     inputs[nb, RFSIZE * RFSIZE + ADDITIONALINPUTS + numactionschosen[nb]] = 1

            # print('inputs: ', inputs.shape)

            inputsC = torch.from_numpy(obs['obs']).to(device)

            ## Running the network
            y, v, (hidden, hebb) = net(inputsC, (hidden, hebb))  # y  should output raw scores, not probas
            # print(v)

            obs, reward, done, info = env.step(
                y
            )

            y = torch.softmax(y, dim=1)
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()  
            logprobs.append(distrib.log_prob(actionschosen))
            numactionschosen = actionschosen.data.cpu().numpy()  # We want to break gradients
            # reward = np.zeros(BATCHSIZE, dtype='float32')

            # ---- Original code run simluation 
            # based on selected action to get reward

            rewards.append(reward.cpu().numpy())
            vs.append(v)
            sumreward += reward

            # This is an "entropy penalty", implemented by the sum-of-squares of the probabilities because our version of PyTorch did not have an entropy() function.
            # The result is the same: to penalize concentration, i.e. encourage diversity in chosen actions.
            loss += ( bent * y.pow(2).sum() / BATCHSIZE )  


            #if PRINTTRACE:
            #    print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
            #            #" - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), 
            #            " -Reward (this step, 1st in batch): ", reward[0])



        # Episode is done, now let's do the actual computations of rewards and losses for the A2C algorithm


        R = torch.zeros(BATCHSIZE).to(device)
        gammaR = gr
        for numstepb in reversed(range(eplen)) :
            R = gammaR * R + torch.from_numpy(rewards[numstepb]).to(device)
            ctrR = R - vs[numstepb][0]
            lossv += ctrR.pow(2).sum() / BATCHSIZE
            loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE  
            #pdb.set_trace()



        loss += blossv * lossv
        loss /= eplen

        if PRINTTRACE:
            if True: #params['algo'] == 'A3C':
                print("lossv: ", float(lossv))
            print ("Total reward for this episode (all):", sumreward, "Dist:", dist)

        loss.backward()
        all_grad_norms.append(torch.nn.utils.clip_grad_norm(net.parameters(), gc))
        if numiter > 100:  # Burn-in period for meanrewards
            optimizer.step()


        lossnum = float(loss)
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward.mean())
            #all_losses_v.append(lossv.data[0])
        #total_loss  += lossnum


        if (numiter+1) % pe == 0:

            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / pe)
            lossbetweensaves = 0
            print("Mean reward (across batch and last", pe, "eps.): ", np.sum(all_total_rewards[-pe:])/ pe)
            #print("Mean reward (across batch): ", sumreward.mean())
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", pe, "iters: ", nowtime - previoustime)
            #print("ETA: ", net.eta.data.cpu().numpy(), " etaet: ", net.etaet.data.cpu().numpy())

        if (numiter+1) % save_every == 0:
            print("Saving files...")
            losslast100 = np.mean(all_losses_objective[-100:])
            print("Average loss over the last 100 episodes:", losslast100)
            print("Saving local files...")
            with open('grad_'+suffix+'.txt', 'w') as thefile:
                for item in all_grad_norms[::10]:
                        thefile.write("%s\n" % item)
            with open('loss_'+suffix+'.txt', 'w') as thefile:
                for item in all_total_rewards[::10]:
                        thefile.write("%s\n" % item)
            torch.save(net.state_dict(), 'torchmodel_'+suffix+'.dat')
            with open('params_'+suffix+'.dat', 'wb') as fo:
                pickle.dump(cfg_dict, fo)
            if os.path.isdir('/mnt/share/tmiconi'):
                print("Transferring to NFS storage...")
                for fn in ['params_'+suffix+'.dat', 'loss_'+suffix+'.txt', 'torchmodel_'+suffix+'.dat']:
                    result = os.system(
                        'cp {} {}'.format(fn, '/mnt/share/tmiconi/modulmaze/'+fn))
                print("Done!")
    #pw = net.initialZeroPlasticWeights()  # For eligibility traces

    #celoss = torch.nn.CrossEntropyLoss() # For supervised learning - not used here

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

    # print('trainable parameters: ', len(init_params))
    # print("Observation space is", env.observation_space)
    # print("Action space is", env.action_space)
    # # print("Action space is", env.action_space)
    # # obs = env.reset()

    # # obs_cpu = obs['obs'].cpu().numpy()
    # # print("Observation: ", obs)
    # actions = torch.ones(cfg.num_envs, ARCHITECTURE[-1])
    # total_rewards = torch.zeros(cfg.num_envs)
    # # total_rewards = torch.unsqueeze(total_rewards, 0)
    # total_rewards = total_rewards.cuda()
    # # print("Action: ", actions)
    # # for i in range(cfg.num_envs):
    # #     actions[i] = models[i].forward(obs_cpu[i])
    # # actions = torch.unsqueeze(actions, 1)
    # # print("Action2: ", actions)
    # actions = actions.cuda()
    # # print("Action2_cuda: ", actions)

    # pop_mean_curve = np.zeros(EPOCHS)
    # best_sol_curve = np.zeros(EPOCHS)
    # eval_curve = np.zeros(EPOCHS)


    # if ARCHITECTURE_NAME == 'Feedforward':
    #     dir_path = './data/'+TASK+'/model/FF/'
    # elif ARCHITECTURE_NAME == 'Hebb':
    #     dir_path = './data/'+TASK+'/model/Hebb/'
    # elif ARCHITECTURE_NAME == 'rbf':
    #     dir_path = 'data/'+TASK+'/model/rbf/'
    # elif ARCHITECTURE_NAME == 'Hebb_rbf':
    #     dir_path = 'data/'+TASK+'/model/Hebb_rbf/'
    # # res = listdir(dir_path+'test_hebb_params/')
    # res = listdir(dir_path)
    # if USE_TRAIN_PARAMS:
    #     for i, file_name in enumerate(res[0:1]):
    #         file_name = 'Hebb_rbf_dbAlpha_rew_puhh_RBFHebb_new_ContactSensor_d_66240499_221.92141723632812.pickle'
    #         print('file_name: ', file_name)
    #         trained_data = pickle.load(open(dir_path+file_name, 'rb'))
    #         open_es_data = trained_data[0]
    #         init_params = open_es_data.best_param() # best_mu   
    #         solver = open_es_data
    #         solver.set_mu(init_params)

    # TEST = cfg.test
    # if TEST == True:
    #     for i, file_name in enumerate(sorted(res)):
    #         # file_name = 'Hebb_rbf_dbAlpha_rew_puhh_RBFHebb_new_ContactSensor_d_66240499_221.92141723632812.pickle'
    #         print('file_name: ', file_name)

    #         # Load Data script
    #         # time.sleep(2)
    #         trained_data = pickle.load(open(dir_path+file_name, 'rb'))
    #         open_es_data = trained_data[0]
    #         # init_params = open_es_data.best_mu # best_mu   
    #         init_params = open_es_data.best_param() # best_mu   
                
    #         # print('trained_data: ', trained_data)
    #         # print('init_params: ', init_params)
    #         # models.set_params_single_model(init_params)            
    #         # models = [None] * cfg.num_envs
    #         # for i in range(cfg.num_envs):
    #         #     models[i] = FeedForwardNet(ARCHITECTURE)
    #         #     models[i].set_params(solutions[i])
    #         # solutions = open_es_data.ask()
    #         obs = env.reset()
            
    #         models.set_params_single_model(init_params)

    #         total_rewards = torch.zeros(cfg.num_envs)
    #         total_rewards = total_rewards.cuda()

    #         obs = env.reset()

    #         for _ in range(EPISODE_LENGTH):
    #             # print('step: ', _)
    #             ############### CPU Version ###############
    #             # actions = models.forward(obs['obs'])
    #             # print('actions: ', actions)
    #             ###########################################
    #             actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #             # print("Action_3: ", actions)
    #             obs, reward, done, info = env.step(
    #                 actions
    #             )
    #             ###########################################
    #             ############### CPU Version Multiple models ###############
    #             # obs_cpu = obs['obs'].cpu().numpy()
    #             # for i in range(cfg.num_envs):
    #             #     actions[i] = models[i].forward(obs_cpu[i])
    #             ##########################################################

    #             ############### GPU Version ###############
    #             # for i in range(cfg.num_envs):
    #             #     actions[i] = init_net.forward(obs['obs'][i])
    #             ###########################################
    #             # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)                # print("Action3: ", actions)

    #             total_rewards += reward

    #         # print("reward is", reward)
    #         print('total_rewards: ', total_rewards)

    # else:
    #     initial_time = timeit.default_timer()
    #     print("initial_time", initial_time)
    #     time_per_epoch_array = []

    #     for epoch in range(EPOCHS):
    #         # print('Epoch: ', epoch)
    #         start_time = timeit.default_timer()
    #         run = 'd'

    #         # retrieve solution from ES 
    #         solutions = solver.ask()
            
    #         # set models parameters 
    #         models.set_params(solutions)

    #         total_rewards = torch.zeros(cfg.num_envs)
    #         total_rewards = total_rewards.cuda()

    #         obs = env.reset()
    #         # obs = obs['obs'].cpu().numpy()
            
    #         # for i in range(cfg.num_envs):
    #         #     actions[i] = models[i].forward(obs['obs'][i])
            
    #         for _ in range(EPISODE_LENGTH):
    #             # print('step: ', _)
    #             ############### CPU Version ###############
    #             # obs_cpu = obs['obs'].cpu().numpy()
    #             # # print('obs_cpu', obs_cpu)
    #             # # print("Observation: ", obs)
    #             # for i in range(cfg.num_envs):
    #             #     actions[i] = models[i].forward(obs_cpu[i])
    #             ###########################################
    #             ############### GPU Version ###############
    #             # obs = torch.zeros(POPSIZE, ARCHITECTURE[0]).cuda()
    #             actions = models.forward(obs['obs'])
    #             # print('actions: ', actions[0, 0:6])
    #             ###########################################
    #             # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #             # print("Action_3: ", actions)
    #             obs, reward, done, info = env.step(
    #                 actions
    #             )
    #             # if env._world.is_playing():
    #             #     if env._world.current_time_step_index == 0:
    #             #         env._world.reset(soft=True)
    #             #     actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #             #     env._task.pre_physics_step(actions)
    #             #     env._world.step(render=render)
    #             #     env.sim_frame_count += 1
    #             #     env._task.post_physics_step()
    #             # else:
    #             #     env._world.step(render=render)
    #             # print('reward: ', reward)
    #             # print('total_rewards: ', total_rewards)
    #             total_rewards += reward

    #         # print("reward is", reward)
    #         # print('total_rewards: ', total_rewards)

    #         total_rewards_cpu = total_rewards.cpu().numpy()
    #         fitlist = list(total_rewards_cpu)
    #         solver.tell(fitlist)

    #         fit_arr = np.array(fitlist)

    #         print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
    #         # with open('log_'+str(run)+'.txt', 'a') as outfile:
    #         #     outfile.write('epoch: ' + str(epoch)
    #         #             + ' mean: ' + str(fit_arr.mean())
    #         #             + ' best: ' + str(fit_arr.max())
    #         #             + ' worst: ' + str(fit_arr.min())
    #         #             + ' std.: ' + str(fit_arr.std()) + '\n')
                
    #         pop_mean_curve[epoch] = fit_arr.mean()
    #         best_sol_curve[epoch] = fit_arr.max()

    #         # WanDB Log data -------------------------------
    #         if wandb_activate:
    #             wandb.log({"epoch": epoch,
    #                         "mean" : np.mean(fitlist),
    #                         "best" : np.max(fitlist),
    #                         "worst": np.min(fitlist),
    #                         "std"  : np.std(fitlist),
    #                         })
    #         # -----------------------------------------------

    #         if (epoch + 1) % SAVE_EVERY == 0:
    #             print('saving..')
    #             pickle.dump((
    #                 solver,
    #                 copy.deepcopy(models),
    #                 pop_mean_curve,
    #                 best_sol_curve,
    #                 ), open(dir_path+exp_name+'_'+str(run)+'_' + str(len(init_params)) + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))
            
    # env._simulation_app.close()




    # # -------------------------------------------------------------------------------
    # # --------- Original code with random policy --------------------------------------
    # # -------------------------------------------------------------------------------
    #     # # simulation GUI config
    #     # headless = cfg.headless
    #     # render = not headless
    #     # enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    #     # # initiate Environment, IsaacGym Simulation
    #     # env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    #     # # sets seed. if seed is -1 will pick a random one
    #     # from omni.isaac.core.utils.torch.maths import set_seed
    #     # cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    #     # cfg_dict['seed'] = cfg.seed
    #     # # initiate Task, Robot
    #     # task = initialize_task(cfg_dict, env)

    #     # # Simulation Loop
    #     # while env._simulation_app.is_running():
    #     #     if env._world.is_playing():
    #     #         if env._world.current_time_step_index == 0:
    #     #             env._world.reset(soft=True)
    #     #         actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #     #         env._task.pre_physics_step(actions)
    #     #         env._world.step(render=render)
    #     #         env.sim_frame_count += 1
    #     #         env._task.post_physics_step()
    #     #     else:
    #     #         env._world.step(render=render)

    #     # env._simulation_app.close()
    # # -------------------------------------------------------------------------------

if __name__ == '__main__':
    parse_hydra_configs()
