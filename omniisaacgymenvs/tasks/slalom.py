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


import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import tensor_clamp, torch_rand_float, unscale
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.stage import get_current_stage
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.slalom import Slalom
from omniisaacgymenvs.tasks.shared.locomotion_simple_rew import LocomotionTask
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema

from omni.isaac.core.prims import RigidPrimView

class SlalomLocomotionTask(LocomotionTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)

        LocomotionTask.__init__(self, name=name, env=env)
        
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_observations = 108
        self._num_actions = 24
        self._ant_positions = torch.tensor([0, 0, 0.0])
        self._terrainType = self._cfg['terrain']['type']
        LocomotionTask.update_config(self)

    def set_up_scene(self, scene) -> None:
        print('-set_up_scene-')

        self._stage = get_current_stage()
        self.get_ant()
        
        # Set up terrain
        if self._terrainType == 'flat':
            RLTask.set_up_scene(self, scene) # Original code without terrain
        else:
            self.get_terrain()
            RLTask.set_up_scene(self, scene, collision_filter_global_paths=["/World/terrain"])
        
        self._ants = ArticulationView(
            prim_paths_expr="/World/envs/.*/geckobotiv/robot_base", name="geckobotiv_view", reset_xform_properties=False
        )
        scene.add(self._ants)

        self.physics_ants = RigidPrimView(
            prim_paths_expr="/World/envs/.*/geckobotiv/robot_base", name="geckobotiv_rigid_view", reset_xform_properties=False
        )
        scene.add(self.physics_ants)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

        indices_ = torch.arange(self._ants.count, dtype=torch.int32, device=self._device)
        max_vel = torch.Tensor(np.full((self._ants.count, self._num_actions), 360.0))
        self._ants.set_max_joint_velocities(max_vel, indices=indices_)

        return

    def initialize_views(self, scene):
        if self._terrainType != 'flat':
            # initialize terrain variables even if we do not need to re-create the terrain mesh
            self.get_terrain(create_mesh=False)

        RLTask.initialize_views(self, scene)
        if scene.object_exists("geckobotiv_view"):
            scene.remove_object("geckobotiv_view", registry_only=True)
        self._ants = ArticulationView(
            prim_paths_expr="/World/envs/.*/geckobotiv/robot_base", name="geckobotiv_view", reset_xform_properties=False
        )
        scene.add(self._ants)

    def get_ant(self):
        ant = Slalom(prim_path=self.default_zero_env_path + "/geckobotiv", name="geckobotiv", translation=self._ant_positions)
        self._sim_config.apply_articulation_settings(
            "geckobotiv", get_prim_at_path(ant.prim_path), self._sim_config.parse_actor_config("geckobotiv")
        )


    def get_robot(self):
        return self._ants

    def post_reset(self):
        self.joint_gears = torch.tensor(np.repeat(8, self._num_actions), dtype=torch.float32, device=self._device)
        dof_limits = self._ants.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self._device)

        force_links = ["pad_lf", "pad_rf", "pad_lh", "pad_rh"]
        self._sensor_indices = torch.tensor(
            [self._ants._body_indices[j] for j in force_links], device=self._device, dtype=torch.long
        )

        LocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self._ants.num_dof)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, num_dof):
    # type: (Tensor, int) -> Tensor
    return torch.sum(obs_buf[:, 12 : 12 + num_dof] > 0.99, dim=-1)