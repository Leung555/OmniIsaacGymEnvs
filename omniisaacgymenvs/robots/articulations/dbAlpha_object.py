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


from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb


class DbAlpha(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "dbAlpha_object",
        # name: Optional[str] = "cartpole",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            dir_path = 'omniverse://localhost/Library/assets/dbalpha/'
            # self._usd_path = dir_path+usd_path+'.usd'

            # Randomization
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_cube_small_1kg_test.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_ball_rd.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_ball_rd_stiff.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_bigball_rd_stiff.usd"

            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_small.usd"
            
            # 1 kg object, heavy
            self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal_1kg.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_cube_small_1kg.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal_tiltL.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal_tiltR.usd"
            
            # 0.1 kg object, light           
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal_0.1kg.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_cube_small_0.1kg.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal_tiltL_0.1kg.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_normal_tiltR_0.1kg.usd"

            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_Nobox.usd"

            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_object_cube_small.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_isaac_test_split_fixTA_instanciable3_5.usd"
            # self._usd_path = "omniverse://localhost/Library/assets/dbalpha/dbAlpha_isaac_test_split_fixTA_instanciable3_5_object.usd"

            print('self._usd_path: ', self._usd_path)

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )
