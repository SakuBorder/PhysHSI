"""Entry point for PhysHSI multi-task transformer training."""

from __future__ import annotations

from TokenHSI.tokenhsi.run import main as token_main
from TokenHSI.tokenhsi.utils import parse_task as token_parse_task_module

from physhsi_amp.env.humanoid_multi_task import HumanoidPhysMultiTask


def _register_physhsi_task() -> None:
    """Inject the PhysHSI multi-task wrapper into the TokenHSI task parser."""

    token_parse_task_module.HumanoidPhysMultiTask = HumanoidPhysMultiTask


def main() -> None:
    """Run the TokenHSI training loop with the PhysHSI task registered."""

    _register_physhsi_task()
    token_main()


if __name__ == "__main__":
    main()
