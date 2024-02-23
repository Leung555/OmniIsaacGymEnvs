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


from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, get_euler_xyz, quat_rotate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

import omni.replicator.isaac as dr

class dbObjectTransportTask(RLTask):
    def __init__(
        self,
        name,
        env,
        offset=None
    ) -> None:

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.position_output_scale = self._task_cfg["env"]["position_output_Scale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]

        self.rew_lin_vel_x_scale = self._task_cfg["env"]["rew_lin_vel_x"]
        self.rew_lin_vel_y_scale = self._task_cfg["env"]["rew_lin_vel_y"]
        self.rew_orient_scale = self._task_cfg["env"]["rew_orient"]
        self.height_reward_scale = self._task_cfg["env"]["height_reward"]
        self.rew_yaw_reward_scale = self._task_cfg["env"]["rew_yaw"]
        self.count = 0
        self.random_joint_initial = True
        self.set_object_RD = False
        print('random_joint_initial: ', self.random_joint_initial)
        print('set_object_RD: ', self.set_object_RD)
        self.joint_index = torch.Tensor([0,1,3,4,5,6,7,8,9,10,11,12])
        # self.rng = np.random.default_rng(12345)

        
        RLTask.__init__(self, name, env)
            
        return

    @abstractmethod
    def set_up_scene(self, scene) -> None:
        pass

    @abstractmethod
    def get_robot(self):
        pass

    def get_observations(self) -> dict:
        # Robot observation
        self.torso_position, self.torso_rotation = self._robots.get_world_poses(clone=False)
        self.velocities = self._robots.get_velocities(clone=False)
        self.velocity = self.velocities[:, 0:3]
        self.ang_velocity = self.velocities[:, 3:6]
        # Robot joint observation
        self.dof_pos = self._robots.get_joint_positions(clone=False)
        self.dof_vel = self._robots.get_joint_velocities(clone=False)

        # object observation
        # self.object_position, self.object_rotation = self._objects.get_world_poses(clone=False)
        # self.object_velocities = self._objects.get_velocities(clone=False)
        # self.object_velocity = self.object_velocities[:, 0:3]

        # Relative Observation for object transportation
        # self.relative_pos = self.object_position - self.torso_position
        # self.relative_yaw = normalize_angle(self.object_rotation[:, 2]) - normalize_angle(self.torso_rotation[:, 2])

        # print('dof_pos: ', self.dof_pos[:, :12])
        # print('dof_vel: ', self.dof_vel)
        # print('object_velocity: ', self.object_velocity)

        roll, pitch, yaw = get_euler_xyz(self.torso_rotation)
        # 6 legs setup
        # self.leg_contact = torch.norm(self._tips.get_net_contact_forces(clone=False).view(self._num_envs, 6, 3), dim=-1) > 0.0
        # 4 legs setup        
        self.leg_contact = torch.norm(self._tips.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1) > 0.0
        # 2 legs setup        
        # self.leg_contact = torch.norm(self._tips.get_net_contact_forces(clone=False).view(self._num_envs, 6, 3), dim=-1) > 0.0
        # print('leg_contact: ', self.leg_contact)

        # self.leg_contact = torch.where(self.leg_contact > 0, 1, -1)        
        self.projected_gravity = quat_rotate(self.torso_rotation, self.gravity_vec)
        # force sensors attached to the feet
        sensor_force_torques = self._robots._physics_view.get_force_sensor_forces() # (num_envs, num_sensors, 6)

        # self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = get_observations(
        #     torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.targets, self.potentials, self.dt,
        #     self.inv_start_rot, self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        #     sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale
        # )
        # to_target = targets - torso_position
        # to_target[:, 2] = 0.0

        # forces = sensor_force_torques[:, :, 0:3]
        # print('forces.shape: ', forces.shape)
        # self.forces_tot = torch.norm(forces, p=2, dim=2)
        # print('forces_tot: ', self.forces_tot[0])
        # print('forces_tot: ', torch.where(forces_tot > 0.02, 1, 0))
        # print('self.leg_contact: ', self.leg_contact[0])

        # print('dof_names: ', self._dbAlphas.dof_names)
        # print(self._tips._prim_paths[0:4])
        # print()
        #prev_potentials = potentials.clone()
        #potentials = -torch.norm(to_target, p=2, dim=-1) / dt

        #torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        #    torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
        #)
#
        #vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        #    torso_quat, velocity, ang_velocity, targets, torso_position
        #)
        # print('orientation: ', [normalize_angle(roll).unsqueeze(-1), 
        #                         normalize_angle(pitch).unsqueeze(-1), 
        #                         normalize_angle(yaw).unsqueeze(-1)])
        # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
        if self.set_object_RD:
            self.pos_observation = torch.cat((self.dof_pos[:, 0:2], self.dof_pos[:, 3:]), dim=1) * 0.5
        else:
            self.pos_observation = self.dof_pos * 0.5

        self.obs_buf = torch.cat(
            (
                # self.pos_observation,
                self.pos_observation,
                self.leg_contact, # 27
                # self.forces_tot, # 27
                # dof_effort, # 39
                normalize_angle(roll).unsqueeze(-1),
                normalize_angle(pitch).unsqueeze(-1),
                normalize_angle(yaw).unsqueeze(-1),
                # input vector for RBF simple modulation
                # self.velocity[:, 0].unsqueeze(-1)

                # Input for object transportation (x,y) axis
                # self.relative_pos[:, 0].unsqueeze(-1),
                # self.relative_pos[:, 1].unsqueeze(-1),
                # self.relative_yaw.unsqueeze(-1)

            ),
            dim=-1,
        )

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        # forces = self.actions * self.joint_gears * self.power_scale
        # target_positions = self.actions
        # print('actions: ', torch.round(actions, decimals=0))

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        # print('self._robots.count: ', self._robots.count)

        # applies joint torques
        # self._robots.set_joint_efforts(forces, indices=indices)

        # applies joint target position
        # self.joint_target_pos = self.actions
        # self.joint_target_pos = torch.cat((self.actions, 
        #                                    self.box_pos), dim=1)
        # print('self.actions_pris: ', self.actions_pris.shape)
        # print('self.box_pos: ', self.box_pos.shape)
        
        # if self.count == 200:
        #     print('reset box pos')
        #     self.box_pos = torch.Tensor(self.num_envs, 1).uniform_(0.0, 0.0)
        # # self.count += 1

        # if self.count == 600:
        #     print('reset box pos')
        #     self.box_pos = torch.Tensor(self.num_envs, 1).uniform_(1.0, 1.0)
        # self.count += 1

        # set box height
        if self.set_object_RD:
            self.actions_pris[:, 0:2] = actions[:, 0:2].clone().to(self._device)
            self.actions_pris[:, 3:] = actions[:, 2:].clone().to(self._device)
            self.actions_pris[:, 2] = self.box_pos.flatten()
            self.joint_target_pos = self.actions_pris
        else:
            self.joint_target_pos = self.actions

        # print('self.box_pos: ', self.joint_target_pos)
        # self.joint_target_pos = 0.5*self.actions + 0.5*self.previous_actions 
        # self.previous_actions = self.actions
        # print(actions)

        self._robots.set_joint_position_targets(self.joint_target_pos, indices=indices)

        if self._dr_randomizer.randomize:
            dr.physics_view.step_randomization(reset_env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # print('num_resets: ', num_resets)

        # randomize DOF positions and velocities
        # print('self._robots.num_dof: ', self._robots.num_dof)
        if self.random_joint_initial == True:
            dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._robots.num_dof), device=self._device)
            dof_pos[:] = tensor_clamp(
                self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper
            )
            dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)
        else:
            dof_pos = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
            # print('dof_pos: ', dof_pos.shape)
            # print('self.box_pos: ', self.box_pos.shape)
            # print('dof_pos[:, 12]: ', dof_pos[:, 12].shape)
            # dof_pos[:, 12] = self.box_pos.flatten()
            dof_vel = torch.zeros((num_resets, self._robots.num_dof), device=self._device)

        # Robot pos
        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        
        # Object pos
        # object_pos, object_rot = self.initial_object_pos[env_ids], self.initial_object_rot[env_ids]
        
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        # Joint
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        # Robot
        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        
        # Object
        # self._objects.set_world_poses(object_pos, object_rot, indices=env_ids)
        # self.random_box_pos = np.random.uniform(-0.1, 0, size=(self.num_envs, 1)).astype(dtype=np.float32)
        box_top_pos = 0.0
        box_low_pos = 0.1
        self.box_pos = torch.Tensor(self.num_envs, 1).uniform_(box_top_pos, box_low_pos)
        # self.box_pos = torch.full((self.num_envs, 1), 1)

        print('box pos: ', box_top_pos, box_low_pos)
        # print('root_pos: ', root_pos)
        self._robots.set_velocities(root_vel, indices=env_ids)

        #to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        #to_target[:, 2] = 0.0
        #self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        #self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self._robots = self.get_robot()

        # Robot Pos
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # Object Pos
        # self.initial_object_pos, self.initial_object_rot = self._objects.get_world_poses()
        
        # initialize some data used later on
        #self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        #self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        #self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        #self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        #self.basis_vec0 = self.heading_vec.clone()
        #self.basis_vec1 = self.up_vec.clone()

        #self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        #self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        #self.dt = 1.0 / 60.0
        #self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        #self.prev_potentials = self.potentials.clone()
        # np.random.seed(-1)
        # self.random_box_pos = np.random.uniform(-0.1, 0, size=(self.num_envs, 1)).astype(dtype=np.float32)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)
        self.actions_pris = torch.zeros((self.num_envs, self.num_actions+1), device=self._device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)
        self.pos_observation = torch.zeros((self.num_envs, self.num_observations), device=self._device)
        # self.box_pos = torch.Tensor(self.num_envs, 1).uniform_(0.0, 0.1).cuda()
        # print('random_box_pos: ', self.box_pos)

        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(
            (self._num_envs, 1)
        )

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def calculate_metrics(self) -> None:
        roll, pitch, yaw = get_euler_xyz(self.torso_rotation)

        # rew_lin_vel_x = self.velocity[:, 0]
        # rew_lin_vel_y = torch.square(self.velocity[:, 1]) * -0.5
        # rew_yaw = torch.square(normalize_angle(yaw)) * -0.5
        # rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -0.5
        # height_reward = torch.square(self.torso_position[:, 2] + 0.1) * -0.5
        # print('rew_lin_vel_x: ', rew_lin_vel_x[0])
        # print('rew_orient: ', rew_orient[0])

        # total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z # + rew_joint_acc + rew_action_rate + rew_cosmetic
        # total_reward = torch.clip(total_reward, 0.0, None)


        # Test lin vel x raward
        rew_lin_vel_x = self.velocity[:, 0] * self.rew_lin_vel_x_scale
        # rew_lin_vel_x_object = self.object_velocity[:, 0] * self.rew_lin_vel_x_scale
        rew_lin_vel_y = torch.square(self.velocity[:, 1]) * -self.rew_lin_vel_y_scale
        rew_orient = torch.where(self.projected_gravity[:, 2] < -0.93 , 0, -self.rew_orient_scale)
        # rew_orient_object = torch.square(self.projected_gravity[:, 2] - 0.866) * -1.0
        height_reward = torch.where(abs(self.torso_position[:, 2] + 0.1) < 0.02 , 0, -self.height_reward_scale)
        rew_yaw = torch.where(abs(normalize_angle(yaw)) < 0.45 , 0, -self.rew_yaw_reward_scale)
        rew_roll = torch.where(abs(normalize_angle(roll)) < 0.3 , 0, -self.rew_yaw_reward_scale)
        # rew_pitch = torch.where(abs(normalize_angle(pitch) - 0.52) < 0.3 , 0, -self.rew_yaw_reward_scale) # ball roll reward
        rew_pitch = torch.where(abs(normalize_angle(pitch)) < 0.3 , 0, -self.rew_yaw_reward_scale)

        gait_reward = torch.ones_like(rew_lin_vel_x) # to get tripod gait Tips idx[2,4,5,0,3,1]
        o1 = torch.zeros_like(rew_lin_vel_x)
        o2 = torch.zeros_like(rew_lin_vel_x)
        # gait_thres = torch.where(self.leg_contact > 0.0, 1, -1)
        # c1 = gait_thres[:, 0] + gait_thres[:, 1] + gait_thres[:, 3]
        # c2 = gait_thres[:, 2] + gait_thres[:, 4] + gait_thres[:, 5]
        # if self.count < 50:
        #     o1 = torch.where(c1 >  2 , 1, 0)
        #     o2 = torch.where(c2 < -2 , 1, 0)
        # elif self.count > 50:
        #     o1 = torch.where(c1 < -2 , 1, 0)
        #     o2 = torch.where(c2 >  2 , 1, 0)

        # gait_reward = (o1 + o2) * 0.3
        # print('gait_reward: ', gait_reward)


        total_reward = rew_lin_vel_x *-1.0 + rew_orient + rew_yaw #+ rew_roll + rew_pitch
        # total_reward = torch.clip(total_reward, 0.0, None)

        # print('rew_lin_vel_x: ', rew_lin_vel_x)
        # print('rew_orient: ', rew_orient)
        # print('height_reward: ', height_reward)
        # print('rew_yaw : ', rew_yaw )

        #self.last_actions[:] = self.actions[:]
        #self.last_dof_vel[:] = self.dof_vel[:]

        # self.fallen_over = self._anymals.is_base_below_threshold(threshold=-0.15, ground_heights=0.0)
        # total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()
        # self.count += 1
        # if self.count > 99:
        #     self.count = 0
    
    def is_done(self) -> None:
        self.reset_buf[:] = is_done(
            self.obs_buf, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


# Original version of calculate_metrics ###################
# @torch.jit.script
# def get_observations(
#     torso_position,
#     torso_rotation,
#     velocity,
#     ang_velocity,
#     dof_pos,
#     dof_vel,
#     targets,
#     potentials,
#     dt,
#     inv_start_rot,
#     basis_vec0,
#     basis_vec1,
#     dof_limits_lower,
#     dof_limits_upper,
#     dof_vel_scale,
#     sensor_force_torques,
#     num_envs,
#     contact_force_scale,
#     actions,
#     angular_velocity_scale
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

#     to_target = targets - torso_position
#     to_target[:, 2] = 0.0

#     prev_potentials = potentials.clone()
#     potentials = -torch.norm(to_target, p=2, dim=-1) / dt

#     torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
#         torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
#     )

#     vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
#         torso_quat, velocity, ang_velocity, targets, torso_position
#     )

#     dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

#     # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
#     obs = torch.cat(
#         (
#             torso_position[:, 2].view(-1, 1),
#             vel_loc,
#             angvel_loc * angular_velocity_scale,
#             normalize_angle(yaw).unsqueeze(-1),
#             normalize_angle(roll).unsqueeze(-1),
#             normalize_angle(angle_to_target).unsqueeze(-1),
#             up_proj.unsqueeze(-1),
#             heading_proj.unsqueeze(-1),
#             dof_pos_scaled,
#             dof_vel * dof_vel_scale,
#             sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
#             actions,
#         ),
#         dim=-1,
#     )

#     return obs, potentials, prev_potentials, up_vec, heading_vec


@torch.jit.script
def is_done(
    obs_buf,
    termination_height,
    reset_buf,
    progress_buf,
    max_episode_length
):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tensor

    # reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return reset


# Original version of calculate_metrics ###################
# @torch.jit.script
# def calculate_metrics(
#     obs_buf,
#     actions,
#     up_weight,
#     heading_weight,
#     potentials,
#     prev_potentials,
#     actions_cost_scale,
#     energy_cost_scale,
#     termination_height,
#     death_cost,
#     num_dof,
#     dof_at_limit_cost,
#     alive_reward_scale,
#     motor_effort_ratio
# ):
#     # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

#     heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
#     heading_reward = torch.where(
#         obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8
#     )

#     # aligning up axis of robot and environment
#     up_reward = torch.zeros_like(heading_reward)
#     up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

#     # energy penalty for movement
#     actions_cost = torch.sum(actions ** 2, dim=-1)
#     electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)

#     # reward for duration of staying alive
#     alive_reward = torch.ones_like(potentials) * alive_reward_scale
#     progress_reward = potentials - prev_potentials

#     total_reward = (
#         progress_reward
#         + alive_reward
#         + up_reward
#         + heading_reward
#         - actions_cost_scale * actions_cost
#         - energy_cost_scale * electricity_cost
#         - dof_at_limit_cost
#     )

#     # adjust reward for fallen agents
#     total_reward = torch.where(
#         obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
#     )
#     return total_reward