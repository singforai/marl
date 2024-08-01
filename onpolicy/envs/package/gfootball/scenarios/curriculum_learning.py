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

class Director():
    def __init__(self):
        self.win_stack  = 0
        self.level = 1


    def assessing_game(self, goal_diffs):
        if all(goal_diff > 0 for goal_diff in goal_diffs):
            self.win_stack += 1
        
        if self.win_stack >= 100:
            self.win_stack = 0
            if self.level < 10:
                self.level += 1

director = Director()

def build_real_scenario(builder):
    builder.config().game_duration = 3000

    builder.config().left_team_difficulty = 1.0
    builder.config().right_team_difficulty = 1.0

    builder.config().deterministic = False
    # if builder.EpisodeNumber() % 2 == 0:
    #     first_team = Team.e_Left
    #     second_team = Team.e_Right
    # else:
    #     first_team = Team.e_Right
    #     second_team = Team.e_Left

    first_team = Team.e_Left
    second_team = Team.e_Right

    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM)
    builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM)
    builder.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)

def build_scenario(builder):
    
    num_agents = 3

    if builder.EpisodeNumber() == 1:
        build_real_scenario(builder)
        return

    difficulty_level = director.level 
    
    if difficulty_level == 10:
        build_real_scenario(builder)
        return

    builder.config().end_episode_on_score = True
    builder.config().game_duration = 3000
    builder.config().left_team_difficulty = 1.0
    builder.config().right_team_difficulty = difficulty_level * 0.1
    builder.config().deterministic = False
    
    # print(dir(builder.config()))
    # if builder.EpisodeNumber() % 2 == 0:
    #   first_team = Team.e_Left
    #   second_team = Team.e_Right
    # else:
    #   first_team = Team.e_Right
    #   second_team = Team.e_Left

    first_team = Team.e_Left
    second_team = Team.e_Right

    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)

    """
    level: 1 => x_pos: 0.4 ~ 0.9

    level: 10 => x_pos: -0.5 ~ 0
    """

    right_margin = 1 - difficulty_level * 0.1 
    left_margin = right_margin - 0.5
    x_pos = list(np.random.random(10) * 0.5 + left_margin)
    y_pos = np.random.normal(loc=0.0, scale=0.1, size=10)
    y_pos = list(np.clip(y_pos, -0.3, 0.3))
     
    player_has_ball = np.random.randint(10)
    builder.SetBallPosition(x_pos[player_has_ball], y_pos[player_has_ball])

    for idx in range(num_agents):
        builder.AddPlayer(x_pos[idx], y_pos[idx], e_PlayerRole_RM)
    # builder.AddPlayer(x_pos[3], y_pos[3], e_PlayerRole_CB)
    # builder.AddPlayer(x_pos[4], y_pos[4], e_PlayerRole_CB)
    # builder.AddPlayer(x_pos[5], y_pos[5], e_PlayerRole_RB)
    # builder.AddPlayer(x_pos[6], y_pos[6], e_PlayerRole_CM)
    # builder.AddPlayer(x_pos[7], y_pos[7], e_PlayerRole_CM)
    # builder.AddPlayer(x_pos[8], y_pos[8], e_PlayerRole_CM)
    # builder.AddPlayer(x_pos[9], y_pos[9], e_PlayerRole_LM)

    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    
    for idx in range(num_agents):
        builder.AddPlayer(x_pos[idx], y_pos[idx], e_PlayerRole_RM)
    # builder.AddPlayer(x_pos[3], y_pos[3], e_PlayerRole_LB)
    # builder.AddPlayer(x_pos[4], y_pos[4], e_PlayerRole_CB)
    # builder.AddPlayer(x_pos[5], y_pos[5], e_PlayerRole_RB)
    # builder.AddPlayer(x_pos[6], y_pos[6], e_PlayerRole_CM)
    # builder.AddPlayer(x_pos[7], y_pos[7], e_PlayerRole_CM)
    # builder.AddPlayer(x_pos[8], y_pos[8], e_PlayerRole_CM)
    # builder.AddPlayer(x_pos[9], y_pos[9], e_PlayerRole_LM)
