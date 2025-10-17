# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from legged_gym.utils.task_registry import task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .g1.carrybox import LeggedRobot as G1CarryBox
from .g1.carrybox_config import G1Cfg as G1CarryBoxCfg
from .g1.carrybox_config import G1CfgPPO as G1CarryBoxCfgPPO

from .g1.carrybox_resume_config import G1Cfg as G1CarryBoxResumeCfg
from .g1.carrybox_resume_config import G1CfgPPO as G1CarryBoxResumeCfgPPO

from .g1.sitdown import LeggedRobot as G1SitDown
from .g1.sitdown_config import G1Cfg as G1SitDownCfg
from .g1.sitdown_config import G1CfgPPO as G1SitDownCfgPPO

from .g1.liedown import LeggedRobot as G1LieDown
from .g1.liedown_config import G1Cfg as G1LieDownCfg
from .g1.liedown_config import G1CfgPPO as G1LieDownCfgPPO

from .g1.standup import LeggedRobot as G1Standup
from .g1.standup_config import G1Cfg as G1StandupCfg
from .g1.standup_config import G1CfgPPO as G1StandupCfgPPO

from .g1.styleloco import LeggedRobot as G1StyleLoco
from .g1.styleloco_dinosaur_config import G1Cfg as G1StyleLocoDinosaurCfg
from .g1.styleloco_dinosaur_config import G1CfgPPO as G1StyleLocoDinosaurCfgPPO
from .g1.styleloco_highknee_config import G1Cfg as G1StyleLocoHighKneeCfg
from .g1.styleloco_highknee_config import G1CfgPPO as G1StyleLocoHighKneeCfgPPO
from .g1.loco_config import G1Cfg as G1LocoCfg
from .g1.loco_config import G1CfgPPO as G1LocoCfgPPO
from .g1.traj import LeggedRobot as G1Traj
from .g1.traj_config import G1Cfg as G1TrajCfg
from .g1.traj_config import G1CfgPPO as G1TrajCfgPPO
from .g1.multi_task import G1MultiTask
from .g1.multi_task_config import G1MultiTaskCfg, G1MultiTaskCfgPPO

task_registry.register( "carrybox", G1CarryBox, G1CarryBoxCfg(), G1CarryBoxCfgPPO() )
task_registry.register( "carrybox_resume", G1CarryBox, G1CarryBoxResumeCfg(), G1CarryBoxResumeCfgPPO() )
task_registry.register( "sitdown", G1SitDown, G1SitDownCfg(), G1SitDownCfgPPO() )
task_registry.register( "liedown", G1LieDown, G1LieDownCfg(), G1LieDownCfgPPO() )
task_registry.register( "standup", G1Standup, G1StandupCfg(), G1StandupCfgPPO() )
task_registry.register( "styleloco_dinosaur", G1StyleLoco, G1StyleLocoDinosaurCfg(), G1StyleLocoDinosaurCfgPPO() )
task_registry.register( "styleloco_highknee", G1StyleLoco, G1StyleLocoHighKneeCfg(), G1StyleLocoHighKneeCfgPPO() )
task_registry.register( "loco", G1StyleLoco, G1LocoCfg(), G1LocoCfgPPO() )
task_registry.register( "traj", G1Traj, G1TrajCfg(), G1TrajCfgPPO() )
task_registry.register( "g1_multi_task", G1MultiTask, G1MultiTaskCfg(), G1MultiTaskCfgPPO() )
