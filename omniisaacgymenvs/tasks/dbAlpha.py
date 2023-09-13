
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


from omniisaacgymenvs.robots.articulations.dbAlpha import DbAlpha
from omniisaacgymenvs.tasks.shared.dbAlpha_locomotion_copy import dbLocomotionTask
from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from pxr import PhysxSchema

import numpy as np
import torch
import math

from omni.isaac.core.prims import RigidPrimView
# from omni.isaac.sensor import _sensor

class dbAlphaLocomotionTask(dbLocomotionTask):
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
        # self._num_observations = 21
        # self._num_actions = 18
        # Leg test
        self._num_observations = 27
        self._num_actions = 18
        self._sim_gear_ratio = 1
        self._dbAlpha_positions = torch.tensor([0, 0, -0.06])
        self._track_contact_forces = True
        self._prepare_contact_sensors = False
        # self._cs = _sensor.acquire_contact_sensor_interface()

        dbLocomotionTask.__init__(self, name=name, env=env)
        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_dbAlpha()
        RLTask.set_up_scene(self, scene)

        self._dbAlphas = ArticulationView(prim_paths_expr="/World/envs/.*/dbAlpha_base", name="dbAlpha_view", reset_xform_properties=False)
        # self._dbAlphas = ArticulationView(prim_paths_expr="/World/envs/.*/dbAlpha", name="robot_view", reset_xform_properties=False, enable_dof_force_sensors=True)
        # self._dbAlphas.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
        scene.add(self._dbAlphas)

        # Add contact force sensor at the robot tips
        self._tips = RigidPrimView(prim_paths_expr="/World/envs/.*/dbAlpha_base/Tips.*",
            name="tips_view", reset_xform_properties=False, 
            track_contact_forces=self._track_contact_forces, 
            prepare_contact_sensors=self._prepare_contact_sensors)
        scene.add(self._tips)

        print('dof_names: ', self._dbAlphas.dof_names)
        print("---------set_up_scene")

        self.leg_contact_bool = torch.zeros((self._num_envs, 6), dtype=torch.float, device=self.device)

        return

    def get_dbAlpha(self):
        dbAlpha = DbAlpha(prim_path=self.default_zero_env_path + "/dbAlpha_base", name="dbAlpha_base", translation=self._dbAlpha_positions)
        self._sim_config.apply_articulation_settings("dbAlpha_base", get_prim_at_path(dbAlpha.prim_path), self._sim_config.parse_actor_config("dbAlpha_base"))
        # dbAlpha = DbAlpha(prim_path=self.default_zero_env_path + "/dbAlpha", name="cartpole", translation=self._dbAlpha_positions)
        # self._sim_config.apply_articulation_settings("dbAlpha", get_prim_at_path(dbAlpha.prim_path), self._sim_config.parse_actor_config("cartpole"))
        # self._sim_config.enable_actor_dof_force_sensors(get_prim_at_path(dbAlpha.prim_path), self._sim_config.parse_actor_config("dbAlpha_base"))
        prim = dbAlpha.prim
        # for link_prim in prim.GetChildren():
        #     if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
        #         rb = PhysxSchema.PhysxRigidBodyAPI.Get(self._stage, link_prim.GetPrimPath())
        #         rb.GetDisableGravityAttr().Set(False)
        #         rb.GetRetainAccelerationsAttr().Set(False)
        #         rb.GetLinearDampingAttr().Set(0.0)
        #         rb.GetMaxLinearVelocityAttr().Set(1000.0)
        #         rb.GetAngularDampingAttr().Set(0.0)
        #         rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "Tips" in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(self._stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)

    def get_robot(self):
        return self._dbAlphas

    def post_reset(self):
        self.joint_gears = torch.tensor(np.repeat(self._sim_gear_ratio, self._num_actions), dtype=torch.float32, device=self._device)
        dof_limits = self._dbAlphas.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self._device)

        dbLocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self._dbAlphas.num_dof)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, num_dof):
    # type: (Tensor, int) -> Tensor
    return torch.sum(obs_buf[:, 0:0+num_dof] > 0.99, dim=-1)