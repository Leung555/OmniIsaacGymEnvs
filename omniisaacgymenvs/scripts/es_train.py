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


import datetime
import os
import gym
import hydra
import torch
from omegaconf import DictConfig
import omniisaacgymenvs
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path, get_experience
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from omniisaacgymenvs.ES.rbf_neural_net import RBFNet
from omniisaacgymenvs.ES.ES_classes import OpenES
import timeit
import pickle
import copy
import numpy as np
# -----------Old code-----------------------------------------------------------------------------
# class RLGTrainer:
#     def __init__(self, cfg, cfg_dict):
#         self.cfg = cfg
#         self.cfg_dict = cfg_dict

#     def launch_rlg_hydra(self, env):
#         # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
#         # We use the helper function here to specify the environment config.
#         self.cfg_dict["task"]["test"] = self.cfg.test

#         # register the rl-games adapter to use inside the runner
#         vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
#         env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env})

#         self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

#     def run(self, module_path, experiment_dir):
#         self.rlg_config_dict["params"]["config"]["train_dir"] = os.path.join(module_path, "runs")

#         # create runner and set the settings
#         runner = Runner(RLGPUAlgoObserver())
#         runner.load(self.rlg_config_dict)
#         runner.reset()

#         # dump config dict
#         os.makedirs(experiment_dir, exist_ok=True)
#         with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
#             f.write(OmegaConf.to_yaml(self.cfg))

#         runner.run(
#             {"train": not self.cfg.test, "play": self.cfg.test, "checkpoint": self.cfg.checkpoint, "sigma": None}
#         )
# ----------------------------------------------------------------------------------------

@hydra.main(version_base=None, config_name="es_config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    # Initialize ES parameters
    POPSIZE             = cfg.num_envs
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
    RBF_ARCHITECTURE = cfg.RBF_ARCHITECTURE
    USE_TRAIN_RBF = cfg.USE_TRAIN_RBF
    
    # Training parameters
    # EPOCHS = configs['Train_params']['EPOCH']
    EPOCHS = cfg.EPOCHS
    EPISODE_LENGTH = cfg.EPISODE_LENGTH
    SAVE_EVERY = cfg.SAVE_EVERY

    # General info
    TASK = cfg.task_name
    TEST = cfg.test
    if TEST:
        USE_TRAIN_RBF = True

    # Initialize model
    if ARCHITECTURE_NAME == 'rbf':
        models = RBFNet(POPSIZE, RBF_ARCHITECTURE[1], RBF_ARCHITECTURE[0], 'loco')
        dir_path = 'runs_ES/'+TASK+'/rbf/'

    n_params_a_model = models.get_n_params_a_model()

    
    # Initialize OpenES Evolutionary Strategy Optimizer
    solver = OpenES(n_params_a_model,
                    popsize=POPSIZE,
                    rank_fitness=RANK_FITNESS,
                    antithetic=ANTITHETIC,
                    learning_rate=LEARNING_RATE,
                    learning_rate_decay=LEARNING_RATE_DECAY,
                    sigma_init=SIGMA_INIT,
                    sigma_decay=SIGMA_DECAY,
                    learning_rate_limit=LEARNING_RATE_LIMIT,
                    sigma_limit=SIGMA_LIMIT)

    # Use train rbf params
    # 1. solver 2. copy.deepcopy(models)  3. pop_mean_curve 4. best_sol_curve,
    if USE_TRAIN_RBF:
        print('--- Used train RBF params ---')
        file_name = 'Ant_80_199_139.4822.pickle'
        print('file_name: ', file_name)
        trained_data = pickle.load(open(dir_path+file_name, 'rb'))
        open_es_data = trained_data[0]
        train_params = open_es_data.best_param() # best_mu   
        print('init_params: ', train_params)
        print('init_params: ', train_params.shape)
        solver.set_mu(train_params)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless

    # local rank (GPU id) in a current multi-gpu mode
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # global rank (GPU id) in multi-gpu multi-node mode
    global_rank = int(os.getenv("RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = local_rank
        cfg.rl_device = f'cuda:{local_rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # select kit app file
    experience = get_experience(headless, cfg.enable_livestream, enable_viewport, cfg.enable_recording, cfg.kit_app)

    env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport or cfg.enable_recording,
        experience=experience
    )

    # parse experiment directory
    module_path = os.path.abspath(os.path.join(os.path.dirname(omniisaacgymenvs.__file__)))
    experiment_dir = os.path.join(module_path, "runs", cfg.train.params.config.name)

    # use gym RecordVideo wrapper for viewport recording
    if cfg.enable_recording:
        if cfg.recording_dir == '':
            videos_dir = os.path.join(experiment_dir, "videos")
        else:
            videos_dir = cfg.recording_dir
        video_interval = lambda step: step % cfg.recording_interval == 0
        video_length = cfg.recording_length
        env.is_vector_env = True
        if env.metadata is None:
            env.metadata = {"render_modes": ["rgb_array"], "render_fps": cfg.recording_fps}
        else:
            env.metadata["render_modes"] = ["rgb_array"]
            env.metadata["render_fps"] = cfg.recording_fps
        env = gym.wrappers.RecordVideo(
            env, video_folder=videos_dir, step_trigger=video_interval, video_length=video_length
        )

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = cfg.seed + global_rank if cfg.seed != -1 else cfg.seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    task = initialize_task(cfg_dict, env)

    if cfg.wandb_activate and global_rank == 0:
        # Make sure to install WandB if you actually use this.
        print()
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            # group=cfg.wandb_group,
            # entity=cfg.wandb_entity,
            config=cfg_dict,
            # sync_tensorboard=False,
            name=run_name,
            # resume="allow",
        )

    torch.cuda.set_device(local_rank)

    # print data on terminal
    print('TASK', TASK)
    print('model: ', ARCHITECTURE_NAME)
    print('model size: ', RBF_ARCHITECTURE)
    print('trainable parameters a model: ', models.get_n_params_a_model())
    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)

    # ------Old code--------------------------------------
    # rlg_trainer = RLGTrainer(cfg, cfg_dict)
    # rlg_trainer.launch_rlg_hydra(env)
    # rlg_trainer.run(module_path, experiment_dir)
    # --------------------------------------------

    # ES code
    # Log data initialize
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)
    
    # initial time to measure time of training loop 
    # initial_time = timeit.default_timer()
    # print("initial_time", initial_time)

    # Testing Loop ----------------------------------
    if TEST:
        # sample params from ES and set model params
        models.set_a_model_params(train_params)
        obs = env.reset()

        # Epoch rewards
        total_rewards = torch.zeros(cfg.num_envs)
        total_rewards = total_rewards.cuda()

        # rollout 
        for _ in range(EPISODE_LENGTH):
            actions = models.forward(obs['obs'])
            obs, reward, done, info = env.step(
                actions
            )
            total_rewards += reward/EPISODE_LENGTH*100


        # update reward arrays to ES
        total_rewards_cpu = total_rewards.cpu().numpy()
        fitlist = list(total_rewards_cpu)
        fit_arr = np.array(fitlist)

        print('mean', fit_arr.mean(), 
            "best", fit_arr.max(), )

    else:
        # Training Loop epoch ###################################
        for epoch in range(EPOCHS):
            # sample params from ES and set model params
            solutions = solver.ask()
            print('solutions: ', solutions.shape)
            models.set_models_params(solutions)
            obs = env.reset()

            # Epoch rewards
            total_rewards = torch.zeros(cfg.num_envs)
            total_rewards = total_rewards.cuda()

            # rollout 
            for _ in range(EPISODE_LENGTH):
                actions = models.forward(obs['obs'])
                obs, reward, done, info = env.step(
                    actions
                )
                total_rewards += reward/EPISODE_LENGTH*100


            # update reward arrays to ES
            total_rewards_cpu = total_rewards.cpu().numpy()
            fitlist = list(total_rewards_cpu)
            solver.tell(fitlist)

            fit_arr = np.array(fitlist)

            print('epoch', epoch, 'mean', fit_arr.mean(), 
                "best", fit_arr.max(), )


            pop_mean_curve[epoch] = fit_arr.mean()
            best_sol_curve[epoch] = fit_arr.max()

            # WanDB Log data -------------------------------
            if cfg.wandb_activate:
                wandb.log({"epoch": epoch,
                            "mean" : np.mean(fitlist),
                            "best" : np.max(fitlist),
                            "worst": np.min(fitlist),
                            "std"  : np.std(fitlist)
                            })
            # -----------------------------------------------

            # Save model params and OpenES params
            if (epoch + 1) % SAVE_EVERY == 0:
                print('saving..')
                pickle.dump((
                    solver,
                    copy.deepcopy(models),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open(dir_path+TASK+'_' + str(n_params_a_model) +'_' + str(epoch) + '_' + str(pop_mean_curve[epoch])[:8] + '.pickle', 'wb'))



    env.close()

    if cfg.wandb_activate and global_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parse_hydra_configs()
