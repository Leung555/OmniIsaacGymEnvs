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

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.dbAlpha2 import DbAlpha
from omniisaacgymenvs.robots.articulations.views.dbAlpha_view import dbAlphaView
# from omniisaacgymenvs.tasks.utils.dbAlpha_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.simulation_context import SimulationContext

import numpy as np
import torch
import math

from pxr import UsdPhysics, UsdLux


class dbAlphaTerrainTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.height_samples = None
        self.custom_origins = False
        self.init_done = False

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self._task_cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]

        #command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.base_threshold = 0.2
        self.knee_threshold = 0.1

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = 27
        self._num_actions = 18

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"]["staticFriction"]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"]["dynamicFriction"]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"]["restitution"]
   
        self._task_cfg["sim"]["add_ground_plane"] = True
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        RLTask.__init__(self, name, env)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.gravity_vec = torch.tensor(get_axis_params(-1., self.up_axis_idx), dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device, requires_grad=False)

        # self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device, requires_grad=False)
        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(), "action_rate": torch_zeros(), "hip": torch_zeros()}
        return


    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:176] = self._task_cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[176:188] = 0. # previous actions
        return noise_vec
    
    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    # def _create_trimesh(self):
    #     self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
    #     vertices = self.terrain.vertices
    #     triangles = self.terrain.triangles
    #     position = torch.tensor([-self.terrain.border_size , -self.terrain.border_size , 0.0])
    #     add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
    #     self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        # self.get_terrain()
        self.get_dbAlpha()
        super().set_up_scene(scene)
        self._dbAlphas = dbAlphaView(prim_paths_expr="/World/envs/.*/dbAlpha", name="dbAlpha_view", track_contact_forces=True)
        scene.add(self._dbAlphas)
        scene.add(self._dbAlphas._tips)
        # scene.add(self._dbAlphas._base)
    


    def get_terrain(self):
        # simple Ground
        pass
        # Complex Curriculum ground
        # self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        # if not self.curriculum: self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        # self.terrain_levels = torch.randint(0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        # self.terrain_types = torch.randint(0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        # self._create_trimesh()
        # self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            
    def get_dbAlpha(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        dbAlpha_translation = torch.tensor([0.0, 0.0, 0.0])
        dbAlpha_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        dbAlpha = DbAlpha(prim_path=self.default_zero_env_path + "/dbAlpha", 
                        name="dbAlpha",
                        translation=dbAlpha_translation, 
                        orientation=dbAlpha_orientation,)
        self._sim_config.apply_articulation_settings("dbAlpha", get_prim_at_path(dbAlpha.prim_path), self._sim_config.parse_actor_config("dbAlpha"))
        dbAlpha.set_dbAlpha_properties(self._stage, dbAlpha.prim)
        dbAlpha.prepare_contacts(self._stage, dbAlpha.prim)

        self.dof_names = dbAlpha.dof_names
        print('self.dof_names: ', self.dof_names)
        print('self.named_default_joint_angles: ', self.named_default_joint_angles)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def post_reset(self):

       # dbAlpha version
       self.initial_root_pos, self.initial_root_rot = self._dbAlphas.get_world_poses()
       self.initial_dof_pos = self._dbAlphas.get_joint_positions()
       dof_limits = self._dbAlphas.get_dof_limits()
       self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
       self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)

       # initialize some data used later on
    #    self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
    #    self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
    #    self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
    #    self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

       # randomize all envs
       indices = torch.arange(self._dbAlphas.count, dtype=torch.int64, device=self._device)
       self.reset_idx(indices)
       # self.init_done = True

        #for i in range(self.num_envs):
        #    self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        # self.num_dof = self._dbAlphas.num_dof
        # self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        # self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        # self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        # self.initial_root_pos, self.initial_root_rot = self._dbAlphas.get_world_poses()        
        # # self.knee_pos = torch.zeros((self.num_envs*4, 3), dtype=torch.float, device=self.device)
        # # self.knee_quat = torch.zeros((self.num_envs*4, 4), dtype=torch.float, device=self.device)
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)
        # self.init_done = True

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._dbAlphas.num_dof), device=self._device)
        dof_pos[:] = tensor_clamp(
            self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper
        )
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._dbAlphas.num_dof), device=self._device)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        self._dbAlphas.set_joint_positions(dof_pos, indices=env_ids)
        self._dbAlphas.set_joint_velocities(dof_vel, indices=env_ids)

        self._dbAlphas.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._dbAlphas.set_velocities(root_vel, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # num_resets = len(env_ids)
# 
    #     indices = env_ids.to(dtype=torch.int32)
    #     positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
    #     velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
    #     self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
    #     self.dof_vel[env_ids] = velocities
    #    # self.update_terrain_level(env_ids)
    #     self.base_pos[env_ids] = self.base_init_state[0:3]
    #     # self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
    #     # self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
    #     self.base_quat[env_ids] = self.base_init_state[3:7]
    #     self.base_velocities[env_ids] = self.base_init_state[7:]
    #     # print('self.base_pos: ', self.base_pos)
    #     # print('self.initial_root_pos: ', self.initial_root_pos)
    #     root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
    #     # print('self.root_pos: ', root_pos)
    #     self._dbAlphas.set_world_poses(positions=self.base_pos[env_ids].clone(), 
    #                                   orientations=self.base_quat[env_ids].clone(),
    #                                   indices=indices)
    #     # self._dbAlphas.set_world_poses(root_pos, root_rot, indices=env_ids)
    #     self._dbAlphas.set_velocities(velocities=self.base_velocities[env_ids].clone(),
    #                                       indices=indices)
    #     self._dbAlphas.set_joint_positions(positions=self.dof_pos[env_ids].clone(), 
    #                                       indices=indices)
    #     self._dbAlphas.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), 
    #                                       indices=indices)
    #    # # self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
    #     # # self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
    #     # # self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
    #     # # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

    #     # # self.last_actions[env_ids] = 0.
    #     # # self.last_dof_vel[env_ids] = 0.
    #     # # self.feet_air_time[env_ids] = 0.
    #     self.progress_buf[env_ids] = 0
    #     self.reset_buf[env_ids] = 0

        # # fill extras
        # self.extras["episode"] = {}
        # for key in self.episode_sums.keys():
        #     self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
        #     self.episode_sums[key][env_ids] = 0.
        # self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
    
    # def update_terrain_level(self, env_ids):
    #     if not self.init_done or not self.curriculum:
    #         # do not change on initial reset
    #         return
    #     root_pos, _ = self._dbAlphas.get_world_poses(clone=False)
    #     distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
    #     self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
    #     self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
    #     self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
    #     self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._dbAlphas.get_joint_positions(clone=False)
        self.dof_vel = self._dbAlphas.get_joint_velocities(clone=False)
    
    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._dbAlphas.get_world_poses(clone=False)
        self.base_velocities = self._dbAlphas.get_velocities(clone=False)
        self.knee_pos, self.knee_quat = self._dbAlphas._tips.get_world_poses(clone=False)

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return

        self.actions = actions.clone().to(self.device)

        self._dbAlphas.set_joint_position_targets(actions)
    
    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self._env._world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            # self.common_step_counter += 1
            # Push robot based on counter
            # if self.common_step_counter % self.push_interval == 0:
            #     self.push_robots()
            
            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        #     forward = quat_apply(self.base_quat, self.forward_vec)
        #     heading = torch.atan2(forward[:, 1], forward[:, 0])
        #     self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.tip_contact = torch.norm(self._dbAlphas._tips.get_net_contact_forces(clone=False).view(self._num_envs, 6, 3), dim=-1) > 0.0
            # print('self.tip_contact: ', self.tip_contact)
            self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))

        #     # print('self.base_quat:', self.base_quat.shape)
        #     # print('self.self.base_velocities:', self.base_velocities.shape)
        #     # print('self.self.base_ang_vel:', self.base_ang_vel.shape)
        #     # print('self.targets:', self.targets.shape)
        #     # print('self.base_pos:', self.base_pos.shape)
            
            self.vel_loc, self.angvel_loc, self.roll, self.pitch, self.yaw, self.angle_to_target = compute_rot(
                self.base_quat, self.base_lin_vel, self.base_ang_vel, self.targets, self.base_pos
            )

            self.check_termination()
            # self.get_states()
            self.calculate_metrics()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            # if self.add_noise:
            #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        #     self.last_actions[:] = self.actions[:]
        #     self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self._dbAlphas.set_velocities(self.base_velocities)
    
    def check_termination(self):
        pass
        # self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
        # knee_contact = torch.norm(self._dbAlphas._tips.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1) > 1.
        # self.has_fallen = (torch.norm(self._dbAlphas._base.get_net_contact_forces(clone=False), dim=1) > 1.) | (torch.sum(knee_contact, dim=-1) > 1.)
        # self.reset_buf = self.has_fallen.clone()
        # self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    def calculate_metrics(self):
        # # velocity tracking reward
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]
        # rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

        # # other base velocity penalties
        # rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        # rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # print('rew_orient: ', rew_orient)

        # base height penalty
        rew_base_height = torch.square(self.base_pos[:, 2] + 0.1)
        # print('rew_base_height: ', rew_base_height)

        # torque penalty
        # rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # joint acc penalty
        # rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        # fallen over penalty
        # rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        # action rate penalty
        # rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # cosmetic penalty for hip motion
        # rew_hip = torch.sum(torch.abs(self.dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1)* self.rew_scales["hip"]

        # Forward Velocity Reward
        lin_vel_rew = torch.square(torch.where(self.base_lin_vel[:, 0] > 0, self.base_lin_vel[:, 0], 0))
        y_lin_vel_rew = torch.square(self.base_lin_vel[:, 1])
        # print('lin_vel_rew: ', lin_vel_rew)
        # print('y_lin_vel_rew: ', y_lin_vel_rew)


        # total reward
        self.rew_buf = (lin_vel_rew
                        - rew_orient * 0.5
                        - rew_base_height * 0.5
                        - y_lin_vel_rew * 0.5
                        )
        # self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # add termination reward
        # self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        # self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        # self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        # self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        # self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        # self.episode_sums["orient"] += rew_orient
        # self.episode_sums["torques"] += rew_torque
        # self.episode_sums["joint_acc"] += rew_joint_acc
        # self.episode_sums["action_rate"] += rew_action_rate
        # self.episode_sums["base_height"] += rew_base_height
        # self.episode_sums["hip"] += rew_hip

    def get_observations(self):
        # self.measured_heights = self.get_heights()
        # heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        # self.tip_contact = torch.zeros(self._num_envs, 6).cuda()
        
        self.obs_buf = torch.cat(
            (  
                self.dof_pos,
                self.tip_contact,
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.pitch).unsqueeze(-1),
                normalize_angle(self.yaw).unsqueeze(-1),                 
            
            ),
            dim=-1)
    
    # def get_ground_heights_below_tips(self):
    #     points = self.knee_pos.reshape(self.num_envs, 4, 3)
    #     points += self.terrain.border_size
    #     points = (points/self.terrain.horizontal_scale).long()
    #     px = points[:, :, 0].view(-1)
    #     py = points[:, :, 1].view(-1)
    #     px = torch.clip(px, 0, self.height_samples.shape[0]-2)
    #     py = torch.clip(py, 0, self.height_samples.shape[1]-2)

    #     heights1 = self.height_samples[px, py]
    #     heights2 = self.height_samples[px+1, py+1]
    #     heights = torch.min(heights1, heights2)
    #     return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    
    # def get_ground_heights_below_base(self):
    #     points = self.base_pos.reshape(self.num_envs, 1, 3)
    #     points += self.terrain.border_size
    #     points = (points/self.terrain.horizontal_scale).long()
    #     px = points[:, :, 0].view(-1)
    #     py = points[:, :, 1].view(-1)
    #     px = torch.clip(px, 0, self.height_samples.shape[0]-2)
    #     py = torch.clip(py, 0, self.height_samples.shape[1]-2)

    #     heights1 = self.height_samples[px, py]
    #     heights2 = self.height_samples[px+1, py+1]
    #     heights = torch.min(heights1, heights2)
    #     return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
                                    
    # def get_heights(self, env_ids=None):
    #     if env_ids:
    #         points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
    #     else:
    #         points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, 0:3]).unsqueeze(1)
 
    #     points += self.terrain.border_size
    #     points = (points/self.terrain.horizontal_scale).long()
    #     px = points[:, :, 0].view(-1)
    #     py = points[:, :, 1].view(-1)
    #     px = torch.clip(px, 0, self.height_samples.shape[0]-2)
    #     py = torch.clip(py, 0, self.height_samples.shape[1]-2)

    #     heights1 = self.height_samples[px, py]

    #     heights2 = self.height_samples[px+1, py+1]
    #     heights = torch.min(heights1, heights2)

    #     return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))
