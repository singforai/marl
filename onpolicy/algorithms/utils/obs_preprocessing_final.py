import numpy as np
import time
import numba

from numba.types import float32, Tuple, bool_, int32

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
    d1_array = np.zeros(d1, dtype = float32)
    for idx, element in enumerate(d2_array):
        d1_array[idx] = np.sqrt(np.sum(np.power(element, 2)))
    return d1_array

@numba.njit(Tuple((float32[:,:], float32[:,:]))(float32[:], int32, int32))
def additive_obs(info, num_agents, episode_length): 
    
    num_teammate = num_agents + 1 
    observation = np.zeros((num_agents, 330), dtype = float32)
    share_observation = np.zeros((num_agents, 220), dtype = float32)

    for agent_idx in range(num_agents):
            
        info_left_team = np.ascontiguousarray(info[10: 32]).reshape(11, 2)
        info_left_direct = np.ascontiguousarray(info[32: 54]).reshape(11, 2)
        info_left_team_tired_factor = info[54: 65]
        info_left_yellow_card = info[65: 76]
        info_left_team_active = info[76: 87] 

        info_right_team = np.ascontiguousarray(info[87: 109]).reshape(11, 2)
        info_right_direct = np.ascontiguousarray(info[109: 131]).reshape(11, 2)
        info_right_tired_factor = info[131: 142]
        info_right_yellow_card = info[142: 153]
        info_right_team_active = info[153: 164]

        info_sticky_actions = np.ascontiguousarray(info[164: 264]).reshape(10, 10)
        info_score = info[264:266]
        
        info_ball = info[266: 269]
        info_ball_direction = info[269: 272]
        info_ball_rotation = info[272: 275]
        info_ball_owned_team = int(info[275])
        info_game_mode = int(info[276])
        info_steps_left = int(info[277])
        info_ball_owned_player = int(info[278]) # left team, right team을 가리지 않고 0:골키퍼 N: 마지막 플레이어까지, -1: 무소유
    

        agent_id = int(agent_idx+ 1)
        # left team 88
        left_position = np.ascontiguousarray(info_left_team).reshape(-1)
        left_direction = np.ascontiguousarray(info_left_direct).reshape(-1)
        left_tired_factor = info_left_team_tired_factor
        left_yellow_card = info_left_yellow_card
        left_red_card = (~np.asarray(info_left_team_active, dtype=bool_)).astype(float32)
        left_offside = np.zeros(11, dtype = float32)
        if info_ball_owned_team == 0:
            left_offside_line = max(0, info_ball[0], np.sort(info_right_team[:num_teammate, 0])[-2])
            left_offside[:num_teammate] = (info_left_team[:num_teammate, 0] > left_offside_line).astype(float32)          
            left_offside[info_ball_owned_player] = 1

        new_left_direction = left_direction.copy()
        
        for counting in range(len(new_left_direction)):
            new_left_direction[counting] = new_left_direction[counting] / direction_x_bound if counting % 2 == 0 else new_left_direction[counting] / direction_y_bound

        left_team = np.zeros(88, dtype =float32)
        left_team[0 : 22] = left_position
        left_team[22: 44] = new_left_direction
        left_team[44: 55] = left_tired_factor
        left_team[55: 66] = left_yellow_card
        left_team[66: 77] = left_red_card
        left_team[77: 88] = left_offside

        # right team 88
        right_position = np.ascontiguousarray(info_right_team).reshape(-1)
        right_direction = np.ascontiguousarray(info_right_direct).reshape(-1)
        right_tired_factor = info_right_tired_factor
        right_yellow_card = info_right_yellow_card
        right_red_card = (~np.asarray(info_right_team_active, dtype=bool_)).astype(float32)
        right_offside = np.zeros(11, dtype = float32)
        if info_ball_owned_team == 1:
            right_offside_line = min(0, info_ball[0], np.sort(info_left_team[:num_teammate, 0])[1])
            right_offside[:num_teammate] = (info_right_team[:num_teammate, 0] < right_offside_line).astype(float32)
            right_offside[info_ball_owned_player] = 1

        new_right_direction = right_direction.copy()
        for counting in range(len(new_right_direction)):
            new_right_direction[counting] = new_right_direction[counting] / direction_x_bound if counting % 2 == 0 else new_right_direction[counting] / direction_y_bound

        right_team = np.zeros(88, dtype =float32)
        right_team[0 : 22] = right_position
        right_team[22: 44] = new_right_direction
        right_team[44: 55] = right_tired_factor
        right_team[55: 66] = right_yellow_card
        right_team[66: 77] = right_red_card
        right_team[77: 88] = right_offside

        # active 18
        active_id = agent_id
        sticky_actions = info_sticky_actions[agent_idx]
        active_position = info_left_team[active_id]
        active_direction = info_left_direct[active_id]
        active_tired_factor = left_tired_factor[active_id]
        active_yellow_card = left_yellow_card[active_id]
        active_red_card = left_red_card[active_id]
        active_offside = left_offside[active_id]

        new_active_direction = active_direction.copy()

        new_active_direction[0] /= direction_x_bound
        new_active_direction[1] /= direction_y_bound
        
        active_player = np.zeros(18, dtype =float32)
        active_player[0 : 10] = sticky_actions
        active_player[10: 12] = active_position
        active_player[12: 14] = new_active_direction
        active_player[14] = active_tired_factor
        active_player[15] = active_yellow_card
        active_player[16] = active_red_card   
        active_player[17] = active_offside  

        relative_ball_position = info_ball[:2] - active_position
        distance2ball = np.linalg.norm(relative_ball_position)
        relative_left_position = info_left_team[:num_teammate, :] - active_position
        distance2left = frobenius_norm_2d(relative_left_position)
        relative_left_position = np.ascontiguousarray(relative_left_position).reshape(-1)
        relative_right_position = info_right_team[:num_teammate, :] - active_position
        distance2right = frobenius_norm_2d(relative_right_position)
        relative_right_position = np.ascontiguousarray(relative_right_position).reshape(-1)

        # relative 69
        relative_info = np.zeros(69, dtype =float32)
        relative_info[0 : 2] = relative_ball_position
        relative_info[2] = distance2ball
        relative_info[3: 3 + 2*num_teammate] = relative_left_position
        relative_info[25: 25 + num_teammate] = distance2left
        relative_info[36: 36 + 2*num_teammate] = relative_right_position
        relative_info[58: 58 + num_teammate] = distance2right
        
        active_info = np.zeros(87, dtype =float32)
        active_info[0:18] = active_player
        active_info[18:87] = relative_info
        # ball info 12
        ball_owned_team = np.zeros(3, dtype =float32)
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
                
        ball_info = np.zeros(12, dtype =float32)
        ball_info[0:3] = info_ball
        ball_info[3:6] = new_ball_direction
        ball_info[6:9] = ball_owned_team
        ball_info[9:12] = new_ball_rotation
        
        # ball owned player 23
        ball_owned_player = np.zeros(23, dtype =float32)
        if info_ball_owned_team == 1: # right_team 
            ball_owned_player[11 + info_ball_owned_player] = 1.0
            ball_owned_player_pos = (info_left_team[info_ball_owned_player]).astype(float32)
            ball_owned_player_direction = (info_right_direct[info_ball_owned_player]).astype(float32)
            ball_owner_tired_factor = right_tired_factor[info_ball_owned_player]
            ball_owner_yellow_card = right_yellow_card[info_ball_owned_player]
            ball_owner_red_card = right_red_card[info_ball_owned_player]
            ball_owner_offside = right_offside[info_ball_owned_player]
        elif info_ball_owned_team == 0: # left_team 
            ball_owned_player[info_ball_owned_player] = 1.0
            ball_owned_player_pos = (info_right_team[info_ball_owned_player]).astype(float32)
            ball_owned_player_direction = (info_left_direct[info_ball_owned_player]).astype(float32)
            ball_owner_tired_factor = left_tired_factor[info_ball_owned_player]
            ball_owner_yellow_card = left_yellow_card[info_ball_owned_player]
            ball_owner_red_card = left_red_card[info_ball_owned_player]
            ball_owner_offside = left_offside[info_ball_owned_player]
        else: # None
            ball_owned_player[-1] = 1.0 
            ball_owned_player_pos = np.zeros(2, dtype = float32)
            ball_owned_player_direction = np.zeros(2, dtype = float32)
        relative_ball_owner_position = np.zeros(2, dtype = float32)

        distance2ballowner = 0
        ball_owner_info = np.zeros(4, dtype = float32)
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

        ball_own_active_info = np.zeros(57, dtype =float32)
        ball_own_active_info[0:12] = ball_info
        ball_own_active_info[12:35] = ball_owned_player
        ball_own_active_info[35:37] = active_position
        ball_own_active_info[37:39] = new_active_direction
        ball_own_active_info[39] = active_tired_factor
        ball_own_active_info[40] = active_yellow_card
        ball_own_active_info[41] = active_red_card
        ball_own_active_info[42] = active_offside
        ball_own_active_info[43: 45] = relative_ball_position
        ball_own_active_info[45] = distance2ball
        ball_own_active_info[46: 48] = ball_owned_player_pos
        ball_own_active_info[48: 50] = new_ball_owned_player_direction
        ball_own_active_info[50: 52] = relative_ball_owner_position
        ball_own_active_info[52] = distance2ballowner
        ball_own_active_info[53: 57] = ball_owner_info

        # match state
        game_mode = np.zeros(7, dtype = float32)
        game_mode[info_game_mode] = 1.0
        goal_diff_ratio = (info_score[0] - info_score[1]) / 5
        steps_left_ratio = info_steps_left / episode_length

        match_state = np.zeros(9, dtype = float32)
        match_state[0:7] = game_mode
        match_state[7] =goal_diff_ratio
        match_state[8] = steps_left_ratio

        observation[agent_idx, 0] = active_id
        observation[agent_idx, 1  : 88] = active_info
        observation[agent_idx, 88 : 145] = ball_own_active_info
        observation[agent_idx, 145: 233] = left_team     
        observation[agent_idx, 233: 321] = right_team 
        observation[agent_idx, 321: 330] = match_state

        share_observation[agent_idx, 0  : 12] = ball_info
        share_observation[agent_idx, 12 : 35] = ball_owned_player
        share_observation[agent_idx, 35 : 123] = left_team     
        share_observation[agent_idx, 123: 211] = right_team 
        share_observation[agent_idx, 211: 220] = match_state  
        
    return (observation, share_observation)

@numba.njit((float32[:,:])(float32[:],float32[:,:],int32[:], int32))
def reward_shaping(info,roll_past_sh_obs, roll_action_env, num_agents):
    added_reward = np.zeros((num_agents, 1), dtype = float32)
    
    info_ball = info[266: 269]
    info_left_team = np.ascontiguousarray(info[10: 32]).reshape(11, 2)
    info_ball_owned_team = int(info[275])
    info_game_mode = int(info[276])
    
    "Passing-Ball reward"
    if info_game_mode == 0:
        if roll_past_sh_obs[0][7] == 1:  # 한 step 전에 우리팀이 공을 소유했는가
            ball_owned_player_idx = np.nonzero(roll_past_sh_obs[0][12:23])[0][0]
            if ball_owned_player_idx != 0:  # 골키퍼가 아닌 agent가 공을 소유하고 있었는가
                if roll_action_env[ball_owned_player_idx - 1] in [9, 10, 11]:  # agent가 pass action을 시도했는가
                    if info_ball_owned_team == 0:  # pass를 한 뒤에도 우리팀이 공을 소유하고 있는가
                        added_reward += 0.05

    "Grouping penalty"
    if info_game_mode == 0:
        for i in range(num_agents):
            for j in range(num_agents):
                if i > j:
                    if np.sqrt(np.sum((info_left_team[i] - info_left_team[j]) ** 2)) < 0.03:
                        added_reward -= 0.001

    "Out-of-bounds penalty"
    if info_game_mode == 0:
        for x_pos, y_pos in info_left_team:
            if np.abs(x_pos) > 1 or np.abs(y_pos) > 0.42:
                added_reward -= 0.001
            
    return added_reward


@numba.njit(Tuple((float32[:,:,:], float32[:,:,:], float32[:,:,:]))(float32[:, :, :],float32[:,:,:],int32[:, :], int32, int32))
def preproc_obs(infos_array,past_share_obs, actions_env, num_agents, episode_length):
    
    observations = np.zeros((infos_array.shape[0], num_agents, 330), dtype = float32)
    share_observations =np.zeros((infos_array.shape[0], num_agents, 220), dtype = float32)
    added_rewards =np.zeros((infos_array.shape[0], num_agents, 1), dtype = float32)
    
    for idx, info_array in enumerate(infos_array):
        added_reward = reward_shaping(
            info = np.ascontiguousarray(info_array).reshape(-1),
            roll_past_sh_obs = past_share_obs[idx], 
            roll_action_env = actions_env[idx],
            num_agents =  num_agents
        )
        obs, share_obs = additive_obs(
            info = np.ascontiguousarray(info_array).reshape(-1),
            num_agents =  num_agents,
            episode_length = episode_length
        )

        observations[idx, : , :] = obs
        share_observations[idx, : , :] = share_obs
        added_rewards[idx, :, :] = added_reward

    return (observations, share_observations, added_rewards)


@numba.njit(Tuple((float32[:,:,:], float32[:,:,:]))(float32[:, :], int32, int32))
def preproc_obs_init(infos_array, num_agents, episode_length):
    
    observations = np.zeros((infos_array.shape[0], num_agents, 330), dtype = float32)
    share_observations =np.zeros((infos_array.shape[0], num_agents, 220), dtype = float32)

    for idx, info_array in enumerate(infos_array):
        obs, share_obs = additive_obs(
            info = np.ascontiguousarray(info_array).reshape(-1),
            num_agents =  num_agents,
            episode_length = episode_length
        )
        observations[idx, : , :] = obs
        share_observations[idx, : , :] = share_obs
    return (observations, share_observations) 


def preprocessing(infos, obs, past_share_obs, actions_env, num_agents, episode_length):
    num_rollout = obs.shape[0]
    num_teammate = num_agents + 1
    
    infos_array = np.zeros((num_rollout, 2, 279), dtype=np.float32)
    for roll_id, info in enumerate(infos):
        info_array = np.zeros(279, dtype = np.float32)
        active = info["active"]
        left_tired_factor = info["left_team_tired_factor"]
        left_team_yellow_card = info["left_team_yellow_card"]
        left_team_active = info["left_team_active"]
        right_tired_factor =info["right_team_tired_factor"]
        right_team_yellow_card = info["right_team_yellow_card"]
        right_team_active = info["right_team_active"]
        sticky_actions = info["sticky_actions"].reshape(-1)
        score = info["score"]
        ball_rotation = info["ball_rotation"]
        left_position = info["left_team"].reshape(-1)
        left_direction = info["left_team_direction"].reshape(-1)
        right_position = info["right_team"].reshape(-1)
        right_direction = info["right_team_direction"].reshape(-1)
        ball_position =info["ball"]
        ball_direction = info["ball_direction"]
        ball_owned_team = info["ball_owned_team"]
        ball_owned_player = info["ball_owned_player"]
        game_mode = info["game_mode"]
        steps_left = info["steps_left"]

        info_array[0  : num_agents] = active
        info_array[10 : 32] = left_position
        info_array[32 : 54] = left_direction
        
        info_array[54 : 54 + num_teammate] = left_tired_factor
        info_array[65 : 65 + num_teammate] = left_team_yellow_card
        info_array[76 : 76 + num_teammate] = left_team_active

        info_array[87  : 109] = right_position
        info_array[109 : 131] = right_direction
        info_array[131 : 131 + num_teammate] = right_tired_factor
        info_array[142 : 142 + num_teammate] = right_team_yellow_card
        info_array[153 : 153 + num_teammate] = right_team_active
            
        info_array[164 : 164 + 10*num_agents] = sticky_actions
        info_array[264 : 266] = score
        info_array[266 : 269] = ball_position
        info_array[269 : 272] = ball_direction
        info_array[272 : 275] = ball_rotation
        info_array[275] = ball_owned_team
        info_array[276] = game_mode
        info_array[277] = steps_left 
        info_array[278] = ball_owned_player
        infos_array[roll_id, :] = info_array
        
    observation, share_observation, added_rewards = preproc_obs(
        infos_array = infos_array, 
        past_share_obs = past_share_obs, 
        actions_env = np.array(actions_env, np.int32), 
        num_agents = num_agents, 
        episode_length = episode_length
    )
    return observation , share_observation, added_rewards

@numba.njit(Tuple((float32[:,:,:], float32[:,:,:]))(float32[:, :, :], int32, int32))
def init_obs(obs , num_agents, episode_length):
    
    num_rollout = obs.shape[0]
    num_teammate = num_agents + 1
    
    obs[:,:,0+2*num_teammate:22] = 0
    obs[:,:,22+2*num_teammate:44] = 0
    obs[:,:,44+2*num_teammate:66] = 0
    obs[:,:,66+2*num_teammate:88] = 0

    team_active = np.zeros(11, dtype = int32)
    team_active[:num_teammate] = 1 

    infos_array = np.zeros((num_rollout, 279), dtype=np.float32)
    for roll_id in range(num_rollout):

        info_array = np.zeros(279, dtype = np.float32)
        
        active = np.arange(1, num_teammate)
        left_position = obs[roll_id,0,0:22]
        left_direction = obs[roll_id,0,22:44]
        left_tired_factor = 0
        left_yellow_card = 0
        left_team_active = team_active
        right_position = obs[roll_id,0,44:66]
        right_direction = obs[roll_id,0,66:88]
        right_tired_factor = 0
        right_yellow_card = 0
        right_team_active = team_active
        
        sticky_actions = 0
        score = 0
        ball_position = obs[roll_id,0,88:91]
        ball_direction = obs[roll_id,0,91:94]
        ball_rotation = 0
        ball_owned_team = np.nonzero(obs[roll_id,0,94:97])[0][0] - 1.0
        game_mode = np.nonzero(obs[roll_id,0,108:115])[0][0]  
        
        info_array[0  : num_agents] = active
        info_array[10 : 32] = left_position
        info_array[32 : 54] = left_direction
        info_array[54 : 54 + num_teammate] = left_tired_factor
        info_array[65 : 65 + num_teammate] = left_yellow_card
        info_array[76 : 87] = left_team_active

        info_array[87  : 109] = right_position
        info_array[109 : 131] = right_direction
        info_array[131 : 131 + num_teammate] = right_tired_factor
        info_array[142 : 142 + num_teammate] = right_yellow_card
        info_array[153 : 164] = right_team_active
            
        info_array[164 : 164 + 10*num_agents] = sticky_actions
        info_array[264 : 266] = score
        info_array[266 : 269] = ball_position
        info_array[269 : 272] = ball_direction
        info_array[272 : 275] = ball_rotation
        info_array[275] = ball_owned_team
        info_array[276] = game_mode
        info_array[277] = episode_length 
        info_array[278] = -1
        infos_array[roll_id, :] = info_array
            
    observation, share_observation = preproc_obs_init(
        infos_array = infos_array, 
        num_agents = num_agents, 
        episode_length = episode_length
    )
    
    return (observation , share_observation) 