# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import json
import os

import numpy as np

from . import *


def build_scenario(builder):
    file_path = "/home/uosai/Desktop/marl/onpolicy/level/level.json"
    if not os.path.exists(file_path):
        file_path = "/home/uosai/Desktop/marl/onpolicy/level/init_level.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            level = json.load(json_file)
    else:
        raise ValueError(f"Error: [json] does not exist.")

    num_agents = level["num_agents"]
    difficulty_level = min(10, level["difficulty_level"])

    builder.config().end_episode_on_score = True
    builder.config().game_duration = 500
    builder.config().left_team_difficulty = 1.0
    builder.config().right_team_difficulty = difficulty_level * 0.1
    builder.config().deterministic = False

    # print(dir(builder.config()))

    first_team = Team.e_Left
    second_team = Team.e_Right

    right_margin = 1 - difficulty_level * 0.1
    left_margin = right_margin - 0.5
    x_pos = list(np.random.random(num_agents) * 0.5 + left_margin)
    y_pos = np.random.normal(loc=0.0, scale=0.1, size=num_agents)
    y_cut = 0.1 + difficulty_level * 0.02
    y_pos = list(np.clip(y_pos, -y_cut, y_cut))

    player_has_ball = np.random.randint(num_agents)
    builder.SetBallPosition(x_pos[player_has_ball], y_pos[player_has_ball])

    position_list = [
        e_PlayerRole_CF,
        e_PlayerRole_LM,
        e_PlayerRole_RM,
        e_PlayerRole_CM,
        e_PlayerRole_LB,
        e_PlayerRole_RB,
        e_PlayerRole_CB,
    ]

    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)

    for idx in range(num_agents):
        position_idx = np.random.randint(len(position_list))
        builder.AddPlayer(x_pos[idx], y_pos[idx], position_list[position_idx])

    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_CB)

    for idx in range(num_agents - 1):
        position_idx = np.random.randint(len(position_list))
        builder.AddPlayer(x_pos[idx], y_pos[idx], position_list[position_idx])
