import numpy as np
import time
import numba

from numba.types import float32, Tuple, bool_, int64

THIRD_X = 0.3
BOX_X = 0.7
MAX_X = 1.0
BOX_Y = 0.24
MAX_Y = 0.42

# Actions.
IDLE = 0
LEFT = 1
TOP_LEFT = 2
TOP = 3
TOP_RIGHT = 4
RIGHT = 5
BOTTOM_RIGHT = 6
BOTTOM = 7
BOTTOM_LEFT = 8
LONG_PASS = 9
HIGH_PASS = 10
SHORT_PASS = 11
SHOT = 12
SPRINT = 13
RELEASE_DIRECTION = 14
RELEASE_SPRINT = 15
SLIDING = 16
DRIBBLE = 17
RELEASE_DRIBBLE = 18
STICKY_LEFT = 0
STICKY_TOP_LEFT = 1
STICKY_TOP = 2
STICKY_TOP_RIGHT = 3
STICKY_RIGHT = 4
STICKY_BOTTOM_RIGHT = 5
STICKY_BOTTOM = 6
STICKY_BOTTOM_LEFT = 7


direction_x_bound = 0.03
direction_y_bound = 0.02
ball_direction_x_bound = 0.15
ball_direction_y_bound = 0.07
ball_direction_z_bound = 4
ball_rotation_x_bound = 0.0005
ball_rotation_y_bound = 0.0004
ball_rotation_z_bound = 0.015


@numba.njit((float32[:])(float32[:, :]))
def frobenius_norm_2d(d2_array):

    d1 = d2_array.shape[0]
    d1_array = np.zeros(d1, dtype=float32)
    for idx, element in enumerate(d2_array):
        d1_array[idx] = np.sqrt(np.sum(np.power(element, 2)))
    return d1_array


@numba.njit(Tuple((float32[:, :], float32[:, :], float32[:, :]))(float32[:], int64, int64))
def thread_processing(info, num_agents, episode_length):  # 279

    num_teammate = num_agents + 1  # goal_keeper

    observation = np.zeros((num_agents, 330), dtype=float32)
    share_observation = np.zeros((num_agents, 220), dtype=float32)
    available_actions = np.ones((num_agents, 19), dtype=float32)

    info_active = info[0:10]  # [1. 2. 3. 0. 0. 0. 0. 0. 0. 0.]

    info_left_team = np.ascontiguousarray(info[10:32]).reshape(11, 2)
    info_left_direct = np.ascontiguousarray(info[32:54]).reshape(11, 2)
    info_left_team_tired_factor = info[54:65]
    info_left_yellow_card = info[65:76]
    info_left_team_active = info[76:87]  # [1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]

    info_right_team = np.ascontiguousarray(info[87:109]).reshape(11, 2)
    info_right_direct = np.ascontiguousarray(info[109:131]).reshape(11, 2)
    info_right_tired_factor = info[131:142]
    info_right_yellow_card = info[142:153]
    info_right_team_active = info[153:164]

    info_sticky_actions = np.ascontiguousarray(info[164:264]).reshape(10, 10)
    info_score = info[264:266]

    info_ball = info[266:269]
    info_ball_direction = info[269:272]
    info_ball_rotation = info[272:275]

    info_ball_owned_team = int(info[275])
    info_game_mode = int(info[276])
    info_steps_left = int(info[277])
    info_ball_owned_player = int(
        info[278]
    )  # left team, right team을 가리지 않고 0:골키퍼 N: 마지막 플레이어까지, -1: 무소유

    RIGHT_ACTIONS = [TOP_RIGHT, RIGHT, BOTTOM_RIGHT, TOP, BOTTOM]
    LEFT_ACTIONS = [TOP_LEFT, LEFT, BOTTOM_LEFT, TOP, BOTTOM]
    BOTTOM_ACTIONS = [BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT, LEFT, RIGHT]
    TOP_ACTIONS = [TOP_LEFT, TOP, TOP_RIGHT, LEFT, RIGHT]

    for idx in range(num_agents):

        agent_id = int(idx + 1)
        # left team 88
        left_position = np.ascontiguousarray(info_left_team).reshape(-1)
        left_direction = np.ascontiguousarray(info_left_direct).reshape(-1)
        left_tired_factor = info_left_team_tired_factor
        left_yellow_card = info_left_yellow_card
        left_red_card = (~np.asarray(info_left_team_active, dtype=bool_)).astype(float32)
        left_offside = np.zeros(11, dtype=float32)
        if info_ball_owned_team == 0:
            left_offside_line = max(0, info_ball[0], np.sort(info_right_team[:num_teammate, 0])[-2])
            left_offside[:num_teammate] = (info_left_team[:num_teammate, 0] > left_offside_line).astype(float32)
            left_offside[info_ball_owned_player] = 1

        new_left_direction = left_direction.copy()

        for counting in range(len(new_left_direction)):
            new_left_direction[counting] = (
                new_left_direction[counting] / direction_x_bound
                if counting % 2 == 0
                else new_left_direction[counting] / direction_y_bound
            )

        left_team = np.zeros(88, dtype=float32)
        left_team[0:22] = left_position
        left_team[22:44] = new_left_direction
        left_team[44:55] = left_tired_factor
        left_team[55:66] = left_yellow_card
        left_team[66:77] = left_red_card
        left_team[77:88] = left_offside

        # right team 88
        right_position = np.ascontiguousarray(info_right_team).reshape(-1)
        right_direction = np.ascontiguousarray(info_right_direct).reshape(-1)
        right_tired_factor = info_right_tired_factor
        right_yellow_card = info_right_yellow_card
        right_red_card = (~np.asarray(info_right_team_active, dtype=bool_)).astype(float32)
        right_offside = np.zeros(11, dtype=float32)
        if info_ball_owned_team == 1:
            right_offside_line = min(0, info_ball[0], np.sort(info_left_team[:num_teammate, 0])[1])
            right_offside[:num_teammate] = (info_right_team[:num_teammate, 0] < right_offside_line).astype(float32)
            right_offside[info_ball_owned_player] = 1

        new_right_direction = right_direction.copy()
        for counting in range(len(new_right_direction)):
            new_right_direction[counting] = (
                new_right_direction[counting] / direction_x_bound
                if counting % 2 == 0
                else new_right_direction[counting] / direction_y_bound
            )

        right_team = np.zeros(88, dtype=float32)
        right_team[0:22] = right_position
        right_team[22:44] = new_right_direction
        right_team[44:55] = right_tired_factor
        right_team[55:66] = right_yellow_card
        right_team[66:77] = right_red_card
        right_team[77:88] = right_offside

        # active 18
        active_id = agent_id
        sticky_actions = info_sticky_actions[idx]
        active_position = info_left_team[active_id]
        active_direction = info_left_direct[active_id]
        active_tired_factor = left_tired_factor[active_id]
        active_yellow_card = left_yellow_card[active_id]
        active_red_card = left_red_card[active_id]
        active_offside = left_offside[active_id]

        new_active_direction = active_direction.copy()

        new_active_direction[0] /= direction_x_bound
        new_active_direction[1] /= direction_y_bound

        active_player = np.zeros(18, dtype=float32)
        active_player[0:10] = sticky_actions
        active_player[10:12] = active_position
        active_player[12:14] = new_active_direction
        active_player[14] = active_tired_factor
        active_player[15] = active_yellow_card
        active_player[16] = active_red_card
        active_player[17] = active_offside

        relative_ball_position = info_ball[:2] - active_position
        distance2ball = np.linalg.norm(relative_ball_position)
        relative_left_position = info_left_team[:num_teammate] - active_position
        distance2left = frobenius_norm_2d(relative_left_position)
        relative_left_position = np.ascontiguousarray(relative_left_position).reshape(-1)
        relative_right_position = info_right_team[:num_teammate] - active_position
        distance2right = frobenius_norm_2d(relative_right_position)
        relative_right_position = np.ascontiguousarray(relative_right_position).reshape(-1)

        # relative 69
        relative_info = np.zeros(69, dtype=float32)
        relative_info[0:2] = relative_ball_position
        relative_info[2] = distance2ball
        relative_info[3 : 3 + 2 * num_teammate] = relative_left_position
        relative_info[25 : 25 + num_teammate] = distance2left
        relative_info[36 : 36 + 2 * num_teammate] = relative_right_position
        relative_info[58 : 58 + num_teammate] = distance2right

        active_info = np.zeros(87, dtype=float32)
        active_info[0:18] = active_player
        active_info[18:87] = relative_info
        # ball info 12
        ball_owned_team = np.zeros(3, dtype=float32)
        ball_owned_team[info_ball_owned_team + 1] = 1.0
        new_ball_direction = info_ball_direction.copy()
        new_ball_rotation = info_ball_rotation.copy()
        for counting in range(len(new_ball_direction)):
            if counting % 3 == 0:
                new_ball_direction[counting] /= ball_direction_x_bound
                new_ball_rotation[counting] /= ball_rotation_x_bound
            if counting % 3 == 1:
                new_ball_direction[counting] /= ball_direction_y_bound
                new_ball_rotation[counting] /= ball_rotation_y_bound
            if counting % 3 == 2:
                new_ball_direction[counting] /= ball_direction_z_bound
                new_ball_rotation[counting] /= ball_rotation_z_bound

        ball_info = np.zeros(12, dtype=float32)
        ball_info[0:3] = info_ball
        ball_info[3:6] = new_ball_direction
        ball_info[6:9] = ball_owned_team
        ball_info[9:12] = new_ball_rotation

        # ball owned player 23
        ball_owned_player = np.zeros(23, dtype=float32)
        if info_ball_owned_team == 1:  # right_team
            ball_owned_player[11 + info_ball_owned_player] = 1.0
            ball_owned_player_pos = (info_left_team[info_ball_owned_player]).astype(float32)
            ball_owned_player_direction = (info_right_direct[info_ball_owned_player]).astype(float32)
            ball_owner_tired_factor = right_tired_factor[info_ball_owned_player]
            ball_owner_yellow_card = right_yellow_card[info_ball_owned_player]
            ball_owner_red_card = right_red_card[info_ball_owned_player]
            ball_owner_offside = right_offside[info_ball_owned_player]
        elif info_ball_owned_team == 0:  # left_team
            ball_owned_player[info_ball_owned_player] = 1.0
            ball_owned_player_pos = (info_right_team[info_ball_owned_player]).astype(float32)
            ball_owned_player_direction = (info_left_direct[info_ball_owned_player]).astype(float32)
            ball_owner_tired_factor = left_tired_factor[info_ball_owned_player]
            ball_owner_yellow_card = left_yellow_card[info_ball_owned_player]
            ball_owner_red_card = left_red_card[info_ball_owned_player]
            ball_owner_offside = left_offside[info_ball_owned_player]
        else:  # None
            ball_owned_player[-1] = 1.0
            ball_owned_player_pos = np.zeros(2, dtype=float32)
            ball_owned_player_direction = np.zeros(2, dtype=float32)

        relative_ball_owner_position = np.zeros(2, dtype=float32)
        distance2ballowner = 0
        ball_owner_info = np.zeros(4, dtype=float32)
        if info_ball_owned_team != -1:
            relative_ball_owner_position = ball_owned_player_pos - active_position
            distance2ballowner = np.linalg.norm(relative_ball_owner_position)

            ball_owner_info[0] = ball_owner_tired_factor
            ball_owner_info[1] = ball_owner_yellow_card
            ball_owner_info[2] = ball_owner_red_card
            ball_owner_info[3] = ball_owner_offside

        new_ball_owned_player_direction = ball_owned_player_direction
        new_ball_owned_player_direction[0] /= direction_x_bound
        new_ball_owned_player_direction[1] /= direction_y_bound

        ball_own_active_info = np.zeros(57, dtype=float32)
        ball_own_active_info[0:12] = ball_info
        ball_own_active_info[12:35] = ball_owned_player
        ball_own_active_info[35:37] = active_position
        ball_own_active_info[37:39] = new_active_direction
        ball_own_active_info[39] = active_tired_factor
        ball_own_active_info[40] = active_yellow_card
        ball_own_active_info[41] = active_red_card
        ball_own_active_info[42] = active_offside
        ball_own_active_info[43:45] = relative_ball_position
        ball_own_active_info[45] = distance2ball
        ball_own_active_info[46:48] = ball_owned_player_pos
        ball_own_active_info[48:50] = new_ball_owned_player_direction
        ball_own_active_info[50:52] = relative_ball_owner_position
        ball_own_active_info[52] = distance2ballowner
        ball_own_active_info[53:57] = ball_owner_info

        # match state
        game_mode = np.zeros(7, dtype=float32)
        game_mode[info_game_mode] = 1.0
        goal_diff_ratio = (info_score[0] - info_score[1]) / 5
        steps_left_ratio = info_steps_left / episode_length

        match_state = np.zeros(9, dtype=float32)
        match_state[0:7] = game_mode
        match_state[7] = goal_diff_ratio
        match_state[8] = steps_left_ratio

        observation[idx, 0] = active_id
        observation[idx, 1:88] = active_info
        observation[idx, 88:145] = ball_own_active_info
        observation[idx, 145:233] = left_team
        observation[idx, 233:321] = right_team
        observation[idx, 321:330] = match_state

        share_observation[idx, 0:12] = ball_info
        share_observation[idx, 12:35] = ball_owned_player
        share_observation[idx, 35:123] = left_team
        share_observation[idx, 123:211] = right_team
        share_observation[idx, 211:220] = match_state

        available_action = np.ones(19, dtype=float32)
        available_action[IDLE] = 0
        available_action[RELEASE_DIRECTION] = 0
        should_left = False

        if game_mode[0] == 1:
            active_x = active_position[0]
            counting_right_enemy_num = 0
            counting_right_teammate_num = 0
            counting_left_teammate_num = 0
            for enemy_pos in info_right_team[1:num_teammate]:
                if active_x < enemy_pos[0]:
                    counting_right_enemy_num += 1
            for teammate_pos in info_left_team[1:num_teammate]:
                if active_x < teammate_pos[0]:
                    counting_right_teammate_num += 1
                if active_x > teammate_pos[0]:
                    counting_left_teammate_num += 1

            if active_x > info_ball[0] + 0.05:
                if counting_left_teammate_num < 2:
                    if info_ball_owned_team != 0:
                        should_left = True

        if should_left:
            available_action = np.zeros(19, dtype=float32)
            for action_idx in RIGHT_ACTIONS:
                available_action[action_idx] = 0
            for action_idx in [LEFT, BOTTOM_LEFT, TOP_LEFT]:
                available_action[action_idx] = 1

            available_action[RELEASE_SPRINT] = 0
            if sticky_actions[8] == 0:
                available_action = np.zeros(19, dtype=float32)
                available_action[SPRINT] = 1

        if abs(relative_ball_position[0]) > 0.75 or abs(relative_ball_position[1]) > 0.5:
            all_directions_vecs = np.zeros((8, 2), dtype=float32)
            ALL_DIRECTION_VECS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
            for v_idx, v in enumerate(ALL_DIRECTION_VECS):
                all_directions_vecs[v_idx, :] = np.array(v) / np.linalg.norm(np.array(v, dtype=float32))
            best_direction = np.zeros(8, dtype=float32)
            for v_idx, v in enumerate(all_directions_vecs):
                best_direction[v_idx] = np.dot(np.ascontiguousarray(relative_ball_position), np.ascontiguousarray(v))
            best_direction = np.argmax(best_direction)
            ALL_DIRECTION_ACTIONS = [LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT]
            target_direction = ALL_DIRECTION_ACTIONS[best_direction]
            forbidden_actions = np.array(ALL_DIRECTION_ACTIONS.copy(), dtype=int64)
            forbidden_actions = forbidden_actions[forbidden_actions != target_direction]
            available_action = np.zeros(19, dtype=float32)
            for action_idx in forbidden_actions:
                available_action[action_idx] = 0
            available_action[target_direction] = 1
            available_action[RELEASE_SPRINT] = 0
            if sticky_actions[8] == 0:
                available_action = np.zeros(19, dtype=float32)
                available_action[SPRINT] = 1

        if_i_hold_ball = info_ball_owned_team == 0 and info_ball_owned_player == active_id
        ball_pos_offset = 0.05
        no_ball_pos_offset = 0.03
        active_x, active_y = active_position[0], active_position[1]
        if_outside = False

        if active_x <= (-1 + no_ball_pos_offset) or (if_i_hold_ball and active_x <= (-1 + ball_pos_offset)):
            if_outside = True
            out_action_index = LEFT_ACTIONS
            target_direction = RIGHT

        elif active_x >= (1 - no_ball_pos_offset) or (if_i_hold_ball and active_x >= (1 - ball_pos_offset)):
            if_outside = True
            out_action_index = RIGHT_ACTIONS
            target_direction = LEFT

        elif active_y >= (0.42 - no_ball_pos_offset) or (if_i_hold_ball and active_y >= (0.42 - ball_pos_offset)):
            if_outside = True
            out_action_index = BOTTOM_ACTIONS
            target_direction = TOP

        elif active_y <= (-0.42 + no_ball_pos_offset) or (if_i_hold_ball and active_x <= (-0.42 + ball_pos_offset)):
            if_outside = True
            out_action_index = TOP_ACTIONS
            target_direction = BOTTOM

        if 1 in game_mode[1:6]:
            left2ball = frobenius_norm_2d(info_left_team[0:num_teammate] - info_ball[:2])
            right2ball = frobenius_norm_2d(info_right_team[0:num_teammate] - info_ball[:2])
            if np.min(left2ball) < np.min(right2ball) and active_id == np.argmin(left2ball):
                if_outside = False

        elif game_mode[6] == 1:
            if info_ball[0] > 0 and active_position[0] > BOX_X:
                if_outside = False

        if if_outside:
            # available_action, sticky_actions, out_action_index, [target_direction], active_direction, False
            available_action = np.zeros(19, dtype=float32)
            for action_idx in out_action_index:
                available_action[action_idx] = 0
            available_action[target_direction] = 1
            available_action[SPRINT] = 0
            if sticky_actions[8] == 1:
                available_action = np.zeros(19, dtype=float32)
                available_action[RELEASE_SPRINT] = 1

        if np.sum(sticky_actions[:8]) == 0:
            available_action[RELEASE_DIRECTION] = 0

        if sticky_actions[8] == 0:
            available_action[RELEASE_SPRINT] = 0
        else:
            available_action[SPRINT] = 0
        if sticky_actions[9] == 0:
            available_action[RELEASE_DRIBBLE] = 0
        else:
            available_action[DRIBBLE] = 0

        if active_position[0] < 0.4 or abs(active_position[1]) > 0.3:
            available_action[SHOT] = 0

        if game_mode[0] == 1:
            if info_ball_owned_team == -1:
                available_action[DRIBBLE] = 0
                if distance2ball >= 0.05:
                    for action_idx in [LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SLIDING]:
                        available_action[action_idx] = 0
            elif info_ball_owned_team == 0:
                available_action[SLIDING] = 0
                if distance2ball >= 0.05:
                    for action_idx in [LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, DRIBBLE]:
                        available_action[action_idx] = 0
            elif info_ball_owned_team == 1:
                available_action[DRIBBLE] = 0
                if distance2ball >= 0.05:
                    for action_idx in [LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SLIDING]:
                        available_action[action_idx] = 0

        elif 1 in game_mode[1:6]:
            left2ball = frobenius_norm_2d(info_left_team[0:num_teammate] - info_ball[:2])
            right2ball = frobenius_norm_2d(info_right_team[0:num_teammate] - info_ball[:2])
            if np.min(left2ball) < np.min(right2ball) and active_id == np.argmin(left2ball):
                for action_idx in [SPRINT, RELEASE_SPRINT, SLIDING, DRIBBLE, RELEASE_DRIBBLE]:
                    available_action[action_idx] = 0
            else:
                for action_idx in [LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SLIDING, DRIBBLE, RELEASE_DRIBBLE]:
                    available_action[action_idx] = 0

        elif game_mode[6] == 1:
            if info_ball[0] > 0 and active_position[0] > BOX_X:
                for action_idx in [
                    LONG_PASS,
                    HIGH_PASS,
                    SHORT_PASS,
                    SPRINT,
                    RELEASE_SPRINT,
                    SLIDING,
                    DRIBBLE,
                    RELEASE_DRIBBLE,
                ]:
                    available_action[action_idx] = 0
            else:
                for action_idx in [LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SLIDING, DRIBBLE, RELEASE_DRIBBLE]:
                    available_action[action_idx] = 0

        available_actions[idx] = available_action
    return (observation, share_observation, available_actions)


@numba.njit(float32[:, :](float32[:], int64))
def reward_shaping(info, num_agents):
    added_reward = np.zeros((num_agents, 1), dtype=float32)
    info_left_team = np.ascontiguousarray(info[10:32]).reshape(11, 2)
    info_left_direct = np.ascontiguousarray(info[32:54]).reshape(11, 2)
    info_left_team_tired_factor = info[54:65]
    info_left_yellow_card = info[65:76]
    info_left_team_active = info[76:87]

    info_right_team = np.ascontiguousarray(info[87:109]).reshape(11, 2)
    info_right_direct = np.ascontiguousarray(info[109:131]).reshape(11, 2)
    info_right_tired_factor = info[131:142]
    info_right_yellow_card = info[142:153]
    info_right_team_active = info[153:164]

    info_sticky_actions = np.ascontiguousarray(info[164:264]).reshape(10, 10)
    info_score = info[264:266]

    info_ball = info[266:269]
    info_ball_direction = info[269:272]
    info_ball_rotation = info[272:275]

    info_ball_owned_team = int(info[275])
    info_game_mode = int(info[276])
    info_steps_left = int(info[277])
    info_ball_owned_player = int(info[278])

    if info_ball_owned_team == 0:
        added_reward += 0.0001

    return added_reward


@numba.njit(
    Tuple((float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :]))(float32[:, :], int64, int64)
)
def preproc_obs(infos_array, num_agents, episode_length):
    observations = np.zeros((infos_array.shape[0], num_agents, 330), dtype=float32)
    share_observations = np.zeros((infos_array.shape[0], num_agents, 220), dtype=float32)
    available_actions = np.zeros((infos_array.shape[0], num_agents, 19), dtype=float32)
    added_rewards = np.zeros((infos_array.shape[0], num_agents, 1), dtype=float32)

    for idx, info_array in enumerate(infos_array):
        added_reward = reward_shaping(info=np.ascontiguousarray(info_array), num_agents=num_agents)
        obs, share_obs, available_action = thread_processing(
            info=np.ascontiguousarray(info_array), num_agents=num_agents, episode_length=episode_length
        )
        observations[idx, :, :] = obs
        share_observations[idx, :, :] = share_obs
        available_actions[idx, :, :] = available_action

    return (
        observations,
        share_observations,
        available_actions,
        added_rewards,
    )  # (num_Rollout, num_agents, 330) / (num_Rollout, num_agents, 220)


def additional_obs(infos, num_agents, episode_length):
    infos_list = []

    num_teammate = num_agents + 1
    num_player = num_teammate * 2

    for info in infos:
        info_array = np.zeros(279, dtype=np.float32)

        info_array[0:num_agents] = info["active"]
        info_array[10 : 10 + num_player] = info["left_team"].reshape(-1)
        info_array[32 : 32 + num_player] = info["left_team_direction"].reshape(-1)
        info_array[54 : 54 + num_teammate] = info["left_team_tired_factor"]
        info_array[65 : 65 + num_teammate] = info["left_team_yellow_card"]
        info_array[76 : 76 + num_teammate] = info["left_team_active"]
        info_array[87 : 87 + num_player] = info["right_team"].reshape(-1)
        info_array[109 : 109 + num_player] = info["right_team_direction"].reshape(-1)
        info_array[131 : 131 + num_teammate] = info["right_team_tired_factor"]
        info_array[142 : 142 + num_teammate] = info["right_team_yellow_card"]
        info_array[153 : 153 + num_teammate] = info["right_team_active"]
        info_array[164 : 164 + 10 * num_agents] = info["sticky_actions"].reshape(-1)

        info_array[264:266] = info["score"]
        info_array[266:269] = info["ball"]
        info_array[269:272] = info["ball_direction"]
        info_array[272:275] = info["ball_rotation"]
        info_array[275] = info["ball_owned_team"]
        info_array[276] = info["game_mode"]
        info_array[277] = info["steps_left"]
        info_array[278] = info["ball_owned_player"]
        infos_list.append(info_array)

    infos_array = np.array(infos_list, dtype=np.float32)
    obs, share_obs, available_actions, added_rewards = preproc_obs(infos_array, num_agents, episode_length)
    return obs, share_obs, available_actions, added_rewards


@numba.njit(Tuple((float32[:, :, :], float32[:, :, :], float32[:, :, :]))(float32[:, :, :], int64, int64))
def init_obs(obs, num_agents, episode_length):
    num_rollout = obs.shape[0]
    num_agents = obs.shape[1]

    left_position = obs[:, :, 0:22]
    left_direction = obs[:, :, 22:44]
    right_position = obs[:, :, 44:66]
    right_direction = obs[:, :, 66:88]
    ball_position = obs[:, :, 88:91]
    ball_direction = obs[:, :, 91:94]
    ball_ownership = obs[:, :, 94:97]
    game_mode = obs[:, :, 108:115]

    left_position[:, :, (num_agents + 1) * 2 : 22] = 0
    left_direction[:, :, (num_agents + 1) * 2 : 22] = 0
    right_position[:, :, (num_agents + 1) * 2 : 22] = 0
    right_direction[:, :, (num_agents + 1) * 2 : 22] = 0

    num_teammate = num_agents + 1

    team_active = np.zeros(11, dtype=int64)
    team_active[:num_teammate] = 1

    obs_array = np.zeros((num_rollout, 279), dtype=np.float32)

    for roll_id in range(num_rollout):
        info_array = np.zeros(279, dtype=np.float32)
        info_array[0:num_agents] = np.arange(1, num_agents + 1)
        info_array[10:32] = np.ascontiguousarray(left_position[roll_id, 0, :]).reshape(-1)
        info_array[32:54] = np.ascontiguousarray(left_direction[roll_id, 0, :]).reshape(-1)
        info_array[54:65] = 0
        info_array[65:76] = 0
        info_array[76:87] = team_active
        info_array[87:109] = np.ascontiguousarray(right_position[roll_id, 0, :]).reshape(-1)
        info_array[109:131] = np.ascontiguousarray(right_direction[roll_id, 0, :]).reshape(-1)
        info_array[131 : 131 + num_teammate] = 0
        info_array[142 : 142 + num_teammate] = 0
        info_array[153:164] = team_active
        info_array[164 : 164 + 10 * num_agents] = 0
        info_array[264:266] = 0
        info_array[266:269] = ball_position[roll_id, 0, :]
        info_array[269:272] = ball_direction[roll_id, 0, :]
        info_array[272:275] = 0  # ball rotation
        info_array[275] = -1  # ball_owned_team
        # info_array[276] = episode_length # steps_left
        info_array[277] = 1  # steps_left
        info_array[278] = -1  # ball_owned_player

        obs_array[roll_id] = info_array

    obs, share_obs, available_actions, _ = preproc_obs(obs_array, num_agents, episode_length)

    return (obs, share_obs, available_actions)
