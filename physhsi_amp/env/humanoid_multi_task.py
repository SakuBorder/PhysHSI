"""PhysHSI-specific multi-task wrapper built on the TokenHSI humanoid stack."""

from __future__ import annotations

import copy
from typing import Any, Dict

from TokenHSI.tokenhsi.env.tasks.multi_task.humanoid_traj_sit_carry_climb import (
    HumanoidTrajSitCarryClimb,
)


class HumanoidPhysMultiTask(HumanoidTrajSitCarryClimb):
    """PhysHSI-focused multi-task variant of the transformer AMP stage.

    The base multi-task humanoid implementation expects canonical task names
    (``traj``, ``sit``, ``carry``, ``climb``).  PhysHSI, however, surfaces the
    skills as ``traj``, ``sitdown``, ``carrybox`` and ``standup``.  This wrapper
    remaps the PhysHSI aliases onto the canonical task identifiers without
    touching the underlying sampling logic.  It also keeps track of the
    user-facing task names so logging and downstream tooling can surface the
    PhysHSI aliases consistently.
    """

    _TASK_NAME_MAPPING: Dict[str, str] = {
        "sitdown": "sit",
        "carrybox": "carry",
        "standup": "climb",
    }

    def __init__(
        self,
        cfg: Dict[str, Any],
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
    ):
        cfg = copy.deepcopy(cfg)
        env_cfg = cfg["env"]

        self._display_names = env_cfg.pop("taskDisplayNames", None)

        env_cfg["task"] = [
            self._TASK_NAME_MAPPING.get(name, name)
            for name in env_cfg.get("task", [])
        ]

        for alias, canonical in self._TASK_NAME_MAPPING.items():
            if alias in env_cfg:
                env_cfg[canonical] = env_cfg.pop(alias)

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        if self._display_names:
            self.extras["each_subtask_name"] = self._display_names
