
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


from omniisaacgymenvs.robots.articulations.dbAlpha_object import DbAlpha
from omniisaacgymenvs.tasks.shared.dbAlpha_object_transport import dbObjectTransportTask
from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from omniisaacgymenvs.tasks.utils.dbAllpha_terrain_generator import *
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from omni.isaac.core.utils.nucleus import get_assets_root_path

from pxr import PhysxSchema

import numpy as np
import torch
import math

from omni.isaac.core.prims import RigidPrimView
# from omni.isaac.sensor import _sensor

class dbAlphaObjectTransportTask(dbObjectTransportTask):
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
        # self._num_observations =  #locomotion_test: 60+6+6+2+6=84
        # self._num_actions = 18
        # Leg test
        self._num_observations = 27 # 27: Jpos, Legcontact, rpy
        self._num_actions = 18
        # 4 legs test
        # self._num_observations = 19 # 27: Jpos, Legcontact, rpy
        # self._num_actions = 12
        self._sim_gear_ratio = 1
        self._dbAlpha_positions = torch.tensor([0, 0, -0.05])
        self._track_contact_forces = True
        self._prepare_contact_sensors = False
        # self._cs = _sensor.acquire_contact_sensor_interface()

        dbObjectTransportTask.__init__(self, name=name, env=env)
        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = get_assets_root_path()
        # self.get_terrain()
        self.get_dbAlpha()
        # self.add_ball()
        self.add_cube()
        # translation = torch.Tensor([0.3, 0.0, 0.1])
        # self.get_object(translation, 0, 0.2)
        RLTask.set_up_scene(self, scene)

        self._dbAlphas = ArticulationView(prim_paths_expr="/World/envs/.*/dbAlpha_base", name="dbAlpha_view", reset_xform_properties=False)
        # self._dbAlphas = ArticulationView(prim_paths_expr="/World/envs/.*/dbAlpha", name="robot_view", reset_xform_properties=False, enable_dof_force_sensors=True)
        # self._dbAlphas.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
        scene.add(self._dbAlphas)

        # self._objects = RigidPrimView(prim_paths_expr="/World/envs/.*/object/object",
        #     name="object_view")  
        self._objects = RigidPrimView(prim_paths_expr="/World/envs/.*/Ball/ball", name="ball_view", reset_xform_properties=False)
        scene.add(self._objects)


        # Add contact force sensor at the robot tips
        self._tips = RigidPrimView(prim_paths_expr="/World/envs/.*/dbAlpha_base/Tips.*",
            name="tips_view", reset_xform_properties=False, 
            track_contact_forces=self._track_contact_forces, 
            prepare_contact_sensors=self._prepare_contact_sensors)
        scene.add(self._tips)

        print("---------set_up_scene")


        # self.leg_contact_bool = torch.zeros((self._num_envs, 6), dtype=torch.float, device=self.device)

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

        # 6 legs setup
        joint_paths = ['base_link/BC1', 'base_link/BC2', 'base_link/BC4', 'base_link/BC5',
                 'dbAlphaL2link1/CF1', 'dbAlphaL2link2/FT1', 'LegAbdomenRearLeftLink1/CF2', 'LegAbdomenRearLeftLink2/FT2', 
                 'LegAbdomenMidRightLink1/CF4', 'LegAbdomenMidRightLink2/FT4',
                 'LegAbdomenRearRightLink1/CF5', 'LegAbdomenRearRightLink2/FT5', 
                 'Thorax/BC0', 'Thorax/BC3',
                 'LegThoraxLeftLink1/CF0', 'LegThoraxLeftLink2/FT0', 
                 'LegThoraxRightLink1/CF3', 'LegThoraxRightLink2/FT3']

        # 4 legs setup
        # joint_paths = ['base_link/BC2', 'base_link/BC5',
        #          'LegAbdomenRearLeftLink1/CF2', 'LegAbdomenRearLeftLink2/FT2', 
        #          'LegAbdomenRearRightLink1/CF5', 'LegAbdomenRearRightLink2/FT5', 
        #          'Thorax/BC0', 'Thorax/BC3',
        #          'LegThoraxLeftLink1/CF0', 'LegThoraxLeftLink2/FT0', 
        #          'LegThoraxRightLink1/CF3', 'LegThoraxRightLink2/FT3']
        # 2 legs setup
        # joint_paths = ['base_link/BC2', 'base_link/BC5',
        #          'LegAbdomenRearLeftLink1/CF2', 'LegAbdomenRearLeftLink2/FT2', 
        #          'LegAbdomenRearRightLink1/CF5', 'LegAbdomenRearRightLink2/FT5']
        for joint_path in joint_paths:
            set_drive(f"{dbAlpha.prim_path}/{joint_path}", "angular", "position", 0, 1, 0.2, 4.1)

        # self.default_dof_pos = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device, requires_grad=False)
        # dof_names = dbAlpha.dof_names
        # for i in range(self.num_actions):
        #     name = dof_names[i]
        #     angle = self.named_default_joint_angles[name]
        #     self.default_dof_pos[:, i] = angle

    def add_ball(self):
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/Ball/ball", 
            translation=[0.4, 0.0, 0.1], 
            name="ball_0",
            radius=0.1,
            color=torch.tensor([0.9, 0.6, 0.2]),
            mass=0.1
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))

    def add_cube(self):
        ball = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/Ball/ball", 
            translation=[0.4, 0.0, 0.1], 
            name="ball_0",
            scale=[0.2, 0.2, 0.2],
            color=torch.tensor([0.9, 0.6, 0.2]),
            mass=0.1
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))

    # def get_object(self, hand_start_translation, pose_dy, pose_dz):
    #     self.object_start_translation = hand_start_translation.clone()
    #     self.object_start_translation[1] += pose_dy
    #     self.object_start_translation[2] += pose_dz
    #     self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
    #     self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
    #     add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
    #     self.object_scale = torch.tensor([5.0, 5.0, 5.0])
    #     # obj = XFormPrim(
    #     #     prim_path=self.default_zero_env_path + "/object/object",
    #     #     name="object",
    #     #     translation=self.object_start_translation,
    #     #     orientation=self.object_start_orientation,
    #     #     scale=self.object_scale,
    #     # )
    #     obj = DynamicCuboid(prim_path="/World/cube_0",
    #         position=np.array([-.5, -.2, 1.0]),
    #         scale=np.array([.5, .5, .5]),
    #         color=np.array([.2,.3,0.])
    #     )
    #     self._sim_config.apply_articulation_settings("object", get_prim_at_path(obj.prim_path), self._sim_config.parse_actor_config("object"))

    def get_terrain(self):
        # self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        # if not self.curriculum: self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        # self.terrain_levels = torch.randint(0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        # self.terrain_types = torch.randint(0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        self._create_trimesh()
        # self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def _create_trimesh(self):
        offset = 120
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size-offset , -self.terrain.border_size-offset , 0.0])
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)  
        # self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def get_robot(self):
        return self._dbAlphas

    def post_reset(self):
        self.joint_gears = torch.tensor(np.repeat(self._sim_gear_ratio, self._num_actions), dtype=torch.float32, device=self._device)
        dof_limits = self._dbAlphas.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self._device)

        dbObjectTransportTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self._dbAlphas.num_dof)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, num_dof):
    # type: (Tensor, int) -> Tensor
    return torch.sum(obs_buf[:, 0:0+num_dof] > 0.99, dim=-1) # remenber to change back to 0+num_dof