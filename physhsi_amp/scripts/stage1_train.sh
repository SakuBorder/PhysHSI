#!/bin/bash

python ./physhsi_amp/run.py --task HumanoidPhysMultiTask \
    --cfg_train physhsi_amp/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_physhsi.yaml \
    --cfg_env physhsi_amp/data/cfg/multi_task/amp_humanoid_physhsi_multi_task.yaml \
    --motion_file TokenHSI/tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs 4096 \
    --headless
