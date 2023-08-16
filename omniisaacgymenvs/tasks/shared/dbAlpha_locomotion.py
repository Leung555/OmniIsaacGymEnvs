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

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
# import omni.isaac.sensor

import numpy as np
import torch
import math

from omni.isaac.core.prims import RigidPrimView

class dbLocomotionTask(RLTask):
    def __init__(
        self,
        name,
        env,
        offset=None
    ) -> None:
        # print("self._task_cfg[env][numEnvs]", self._task_cfg["env"]["numEnvs"])
        # print("_env_spacing", self._task_cfg["env"]["envSpacing"])
        # print("_max_episode_length", self._task_cfg["env"]["episodeLength"])
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
        self.joint_damp = 100000.0
        self.joint_stiffness = 10000000.0
        self.velocity = [0,0,0]
        self.ang_velocity = [0,0,0]
        # self._track_contact_forces = True
        # self._prepare_contact_sensors = False
        
        # Acquire the contact sensor interface
        # self.contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        # Define the sensor path
        # self.contact_sensor_path = "/World/envs/env_0/dbAlpha_base/Tips0/Contact_Sensor_0"

        RLTask.__init__(self, name, env)
        return

    @abstractmethod
    def set_up_scene(self, scene) -> None:
        pass

    @abstractmethod
    def get_robot(self):
        pass

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        self.torso_position = torso_position
        # print('self.torso_position: ', self.torso_position)
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        self.velocity = velocity
        ang_velocity = velocities[:, 3:6]
        self.ang_velocity = ang_velocity
        # self._robots.get_applied_action()
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)
        dof_target = self._robots.get_applied_actions(clone=False)
        dof_names = self._robots.dof_names
        # print('dof_target pos: ', dof_target.joint_positions)
        # print('dof_target vel: ', dof_target.joint_velocities)
        # print('dof_target effort: ', dof_target.joint_efforts)
        # dof_effort = self._robots.get_applied_joint_efforts(clone=False)
        dof_effort = 0.0000001 * (self.joint_stiffness * (dof_target.joint_positions - dof_pos) + self.joint_damp * (dof_target.joint_velocities - dof_vel))
        # print('dof_pos: ', dof_pos)
        # print('dof_vel: ', dof_vel)
        # print('dof_vel: ', dof_vel)
        # print('dof_effort: ', dof_effort)
        # print('dof_names: ', dof_names)

        # force sensors attached to the feet
        sensor_force_torques = self._robots._physics_view.get_force_sensor_forces() # (num_envs, num_sensors, 6)
        # print('sensor_force_torques: ', sensor_force_torques)
        # print('sensor_force_torques: ', sensor_force_torques.shape)
        # self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = get_observations(
        #     torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.targets, self.potentials, self.dt,
        #     self.inv_start_rot, self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        #     sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale
        # )
        
        # leg_contact = torch.norm(self._tips.get_net_contact_forces(clone=False).view(self._num_envs, 5, 3), dim=-1) > 0.
        # print("leg_contact:", leg_contact)
        
        # Run the simulation (make sure to start the simulation before trying to get sensor readings)

        # Get sensor readings
        # sensor_data = self.contact_sensor_interface.get_sensor_readings(self.contact_sensor_path)

        # Print sensor data
        # print("Sensor Data:", sensor_data)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.vel_loc, self.ang_loc, self.up_proj, self.heading_proj = get_observations(
            torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.targets, self.potentials, self.dt,
            self.inv_start_rot, self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale, 
            dof_effort
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
        # print('actions: ', actions)

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        # print('self._robots.count: ', self._robots.count)

        # applies joint torques
        # self._robots.set_joint_efforts(forces, indices=indices)

        # applies joint target position
        self.joint_target_pos = self.actions
        # print(actions)
        self._robots.set_joint_position_targets(self.joint_target_pos, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        # print('self._robots.num_dof: ', self._robots.num_dof)
        dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._robots.num_dof), device=self._device)
        dof_pos[:] = tensor_clamp(
            self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper
        )
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.up_proj = torch.tensor([0], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.heading_proj = torch.tensor([0], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        # print('self.heading_proj: ', self.heading_proj)
        # print('self.up_proj: ', self.up_proj)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # print('self.heading_proj: ', self.heading_proj)
        # print('self.up_proj: ', self.up_proj)        
        self.rew_buf[:] = calculate_metrics(
            self.obs_buf, self.actions, self.up_weight, self.heading_weight, self.potentials, self.prev_potentials,
            self.actions_cost_scale, self.energy_cost_scale, self.termination_height,
            self.death_cost, self._robots.num_dof, self.alive_reward_scale, self.motor_effort_ratio, 
            self.heading_proj, self.up_proj, self.velocity, self.ang_velocity, self.torso_position,
            self.vel_loc, self.ang_loc
        )

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

@torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    targets,
    potentials,
    dt,
    inv_start_rot,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    num_envs,
    contact_force_scale,
    actions,
    angular_velocity_scale,
    dof_effort
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    forces = sensor_force_torques[:, :, 0:3]
    # print('forces.shape: ', forces.shape)
    forces_tot = torch.norm(forces, p=2, dim=2)
    # print('forces_tot: ', forces_tot)
    # print('forces_tot: ', torch.where(forces_tot > 0.02, 1, 0))

    prev_potentials = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    dof_effort  = dof_effort * 0.1
    
    # print('heading_vec: ', heading_vec.shape)
    # print('heading_proj: ', heading_proj.shape)
    # print('heading_proj.unsqueeze(-1): ', heading_proj.unsqueeze(-1).shape)
    # print('up_vec: ', up_vec.shape)
    # print('up_proj: ', up_proj.shape)
    # print('up_proj.unsqueeze(-1): ', up_proj.unsqueeze(-1).shape)
    # print('forces_tot: ', forces_tot)
    # print('forces_tot: ', forces_tot.shape)
    # print('dof_pos_scaled: ', dof_pos_scaled.shape)
    # print('dof_pos: ', dof_pos)
    # print('normalize_angle(roll).unsqueeze(-1): ', normalize_angle(roll))
    # print('normalize_angle(pitch).unsqueeze(-1): ', normalize_angle(pitch))
    # print('normalize_angle(yaw).unsqueeze(-1): ', normalize_angle(yaw))

    obs = torch.cat(
        (
            dof_pos,
            forces_tot, # 27
            # dof_effort, # 39
            normalize_angle(roll).unsqueeze(-1),
            normalize_angle(pitch).unsqueeze(-1),
            normalize_angle(yaw).unsqueeze(-1),
        ),
        dim=-1,
    )

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    # obs = torch.cat(
    #     (
    #         torso_position[:, 2].view(-1, 1),
    #         vel_loc,
    #         angvel_loc * angular_velocity_scale,
    #         normalize_angle(yaw).unsqueeze(-1),
    #         normalize_angle(roll).unsqueeze(-1),
    #         normalize_angle(angle_to_target).unsqueeze(-1),
    #         up_proj.unsqueeze(-1),
    #         heading_proj.unsqueeze(-1),
    #         dof_pos_scaled,
    #         dof_vel * dof_vel_scale,
    #         sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
    #         actions,
    #     ),
    #     dim=-1,
    # )

    return obs, potentials, prev_potentials, vel_loc, angvel_loc, up_proj, heading_proj


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

@torch.jit.script
def calculate_metrics(
    obs_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    alive_reward_scale,
    motor_effort_ratio,
    heading_proj, 
    up_proj,
    velocity, 
    ang_velocity,
    torso_position,
    vel_loc,
    angvel_loc
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    # heading_proj = heading_proj.unsqueeze(-1)
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(
        heading_proj > 0.9, 0, -heading_weight_tensor
        # heading_proj > 0.8, heading_weight_tensor, -0.5
    )

    # aligning up axis of robot and environment
    up_reward = torch.ones_like(heading_reward) * up_weight
    up_reward = torch.where(up_proj > 0.93, 0, -up_reward)

    height_reward = torch.ones_like(heading_reward) * 0.2
    height_reward = torch.where(abs(torso_position[:, 2] + 0.1) < 0.02 , 0, -height_reward)

    # energy penalty for movement
    actions_cost = torch.mean(torch.abs(actions), dim=-1)
    actions_cost = torch.where(actions_cost < 0.52, 0, -0.2)
    # electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials


    # print('progress_reward: ', progress_reward)
    # print('heading_reward: ', heading_reward)
    # print('up_reward: ', up_reward)
    # print('height_reward: ', height_reward)
    # print('torso_position[:, 2]: ', torso_position[:, 2])
    # print('velocity: ', velocity)
    # print('ang_velocity: ', ang_velocity)
    # print('vel_loc: ', vel_loc)
    # print('angvel_loc: ', angvel_loc)
    # print('actions_cost: ', actions_cost[0:5])

    total_reward = (
        progress_reward*4
        # vel_loc[:, 0] * 16.0
        # - abs(angvel_loc[:, 0])
        # - abs(angvel_loc[:, 1])
        # - abs(angvel_loc[:, 2])
        # torch.abs(ang_velocity[:, 0])
        # - torch.abs(velocity[:, 1])
        # - torch.abs(velocity[:, 2])
        + up_reward *2
        + heading_reward  *2
        + height_reward *2
        # + actions_cost *2
    )   

    # total_reward = (
    #     progress_reward
    #     + alive_reward
    #     + up_reward
    #     + heading_reward
    #     - actions_cost_scale * actions_cost
    #     - energy_cost_scale * electricity_cost
    #     - dof_at_limit_cost
    # )

    # adjust reward for fallen agents
    # total_reward = torch.where(
    #     obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    # )
    return total_reward

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