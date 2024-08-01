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

RIGHT_ACTIONS = [TOP_RIGHT, RIGHT, BOTTOM_RIGHT, TOP, BOTTOM]
LEFT_ACTIONS = [TOP_LEFT, LEFT, BOTTOM_LEFT, TOP, BOTTOM]
BOTTOM_ACTIONS = [BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT, LEFT, RIGHT]
TOP_ACTIONS = [TOP_LEFT, TOP, TOP_RIGHT, LEFT, RIGHT]
ALL_DIRECTION_ACTIONS = [LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT]
ALL_DIRECTION_VECS = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


direction_x_bound = 0.03
direction_y_bound = 0.02
ball_direction_x_bound = 0.15
ball_direction_y_bound = 0.07
ball_direction_z_bound = 4
ball_rotation_x_bound = 0.0005
ball_rotation_y_bound = 0.0004
ball_rotation_z_bound = 0.015

@numba.njit(Tuple((float32[:,:], float32[:,:]))(float32[:], int64))
def thread_processing(info, num_agents): # 279
    
    observation = np.zeros((num_agents, 330), dtype = float32)
    share_observation = np.zeros((num_agents, 220), dtype = float32)

    info_active = info[0:10]

    info_left_team = np.ascontiguousarray(info[10: 32]).reshape(11, 2)
    info_left_direct = np.ascontiguousarray(info[32: 54]).reshape(11, 2)
    info_left_team_tired_factor = info[54: 65]
    info_left_yellow_card = info[65: 76]
    info_left_team_active = info[76:87]

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
    info_ball_owned_player = int(info[278])

    for idx, agent_id in enumerate(info_active):
        if agent_id == -1:
            break
        agent_id = int(agent_id)
        # left team 88
        left_position = np.ascontiguousarray(info_left_team).reshape(-1)
        left_direction = np.ascontiguousarray(info_left_direct).reshape(-1)
        left_tired_factor = info_left_team_tired_factor
        left_yellow_card = info_left_yellow_card
        left_red_card = (~np.asarray(info_left_team_active, dtype=bool_)).astype(float32)
        left_offside = np.zeros(11, dtype = float32)
        if info_ball_owned_team == 0:
            left_offside_line = max(0, info_ball[0], np.sort(info_right_team[:, 0])[-2])
            left_offside = (info_left_team[:, 0] > left_offside_line).astype(float32) 
            """
            아래의 코드를 변경함. 
            left_offside[info_ball_owned_player] = False => left_offside[info_ball_owned_player] = 1
            """
            left_offside[info_ball_owned_player] = 1

        new_left_direction = left_direction.copy()
        for counting in range(len(new_left_direction)):
            new_left_direction[counting] = new_left_direction[counting] / direction_x_bound if counting % 2 == 0 else new_left_direction[counting] / direction_y_bound

        left_team = np.zeros(88)
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
            right_offside_line = min(0, info_ball[0], np.sort(info_left_team[:, 0])[1])
            right_offside = (info_right_team[:, 0] < right_offside_line).astype(float32)
            right_offside[info_ball_owned_player] = 1

        new_right_direction = right_direction.copy()
        for counting in range(len(new_right_direction)):
            new_right_direction[counting] = new_right_direction[counting] / direction_x_bound if counting % 2 == 0 else new_right_direction[counting] / direction_y_bound

        right_team = np.zeros(88)
        right_team[0 : 22] = right_position
        right_team[22: 44] = new_right_direction
        right_team[44: 55] = right_tired_factor
        right_team[55: 66] = right_yellow_card
        right_team[66: 77] = right_red_card
        right_team[77: 88] = right_offside

        # active 18
        active_id = agent_id
        sticky_actions = info_sticky_actions[idx]
        active_position = info_left_team[agent_id]
        active_direction = info_left_direct[agent_id]
        active_tired_factor = left_tired_factor[agent_id]
        active_yellow_card = left_yellow_card[agent_id]
        active_red_card = left_red_card[agent_id]
        active_offside = left_offside[agent_id]

        new_active_direction = active_direction.copy()

        new_active_direction[0] /= direction_x_bound
        new_active_direction[1] /= direction_y_bound
        
        active_player = np.zeros(18)
        active_player[0 : 10] = sticky_actions
        active_player[10: 12] = active_position
        active_player[12: 14] = new_active_direction
        active_player[14] = active_tired_factor
        active_player[15] = active_yellow_card
        active_player[16] = active_red_card   
        active_player[17] = active_offside  

        relative_ball_position = info_ball[:2] - active_position
        distance2ball = np.linalg.norm(relative_ball_position)
        relative_left_position = info_left_team - active_position
        distance2left = np.linalg.norm(relative_left_position, 1)
        relative_left_position = np.ascontiguousarray(relative_left_position).reshape(-1)
        relative_right_position = info_right_team - active_position
        distance2right = np.linalg.norm(relative_right_position, 1)
        relative_right_position = np.ascontiguousarray(relative_right_position).reshape(-1)

        # relative 69
        relative_info = np.zeros(69)
        relative_info[0 : 2] = relative_ball_position
        relative_info[2] = distance2ball
        relative_info[3: 25] = relative_left_position
        relative_info[25: 36] = distance2left
        relative_info[36: 58] = relative_right_position
        relative_info[58: 69] = distance2right

        active_info = np.zeros(87)
        active_info[0:18] = active_player
        active_info[18:87] = relative_info
        
        # ball info 12
        ball_owned_team = np.zeros(3)
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
                
        ball_info = np.zeros(12)
        ball_info[0:3] = info_ball
        ball_info[3:6] = new_ball_direction
        ball_info[6:9] = ball_owned_team
        ball_info[9:12] = new_ball_rotation

        # ball owned player 23
        ball_owned_player = np.zeros(23)
        if info_ball_owned_team == 1:     # 对手
            ball_owned_player[11 + info_ball_owned_player] = 1.0
            ball_owned_player_pos = (info_left_team[info_ball_owned_player]).astype(float32)
            ball_owned_player_direction = (info_right_direct[info_ball_owned_player]).astype(float32)
            ball_owner_tired_factor = right_tired_factor[info_ball_owned_player]
            ball_owner_yellow_card = right_yellow_card[info_ball_owned_player]
            ball_owner_red_card = right_red_card[info_ball_owned_player]
            ball_owner_offside = right_offside[info_ball_owned_player]
        elif info_ball_owned_team == 0:
            ball_owned_player[info_ball_owned_player] = 1.0
            ball_owned_player_pos = (info_right_team[info_ball_owned_player]).astype(float32)
            ball_owned_player_direction = (info_left_direct[info_ball_owned_player]).astype(float32)
            ball_owner_tired_factor = left_tired_factor[info_ball_owned_player]
            ball_owner_yellow_card = left_yellow_card[info_ball_owned_player]
            ball_owner_red_card = left_red_card[info_ball_owned_player]
            ball_owner_offside = left_offside[info_ball_owned_player]
        else:
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

        ball_own_active_info = np.zeros(57)
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
        steps_left_ratio = info_steps_left / 3001

        match_state = np.zeros(9, dtype = float32)
        match_state[0:7] = game_mode
        match_state[7] =goal_diff_ratio
        match_state[8] = steps_left_ratio

        agent_obs = np.zeros((1 ,330), dtype = float32)
        agent_obs[:, 0] = active_id
        agent_obs[:,1  : 88] = active_info
        agent_obs[:,88 : 145] = ball_own_active_info
        agent_obs[:,145: 233] = left_team     
        agent_obs[:,233: 321] = right_team 
        agent_obs[:,321: 330] = match_state

        observation[idx, :] = agent_obs

        share_obs = np.zeros((1, 220), dtype = float32)
        share_obs[ : , 0  : 12] = ball_info
        share_obs[ : , 12 : 35] = ball_owned_player
        share_obs[ : , 35 : 123] = left_team     
        share_obs[ : , 123: 211] = right_team 
        share_obs[ : , 211: 220] = match_state  

        share_observation[idx, :] = share_obs
    
    return (observation, share_observation)


@numba.njit(Tuple((float32[:,:,:], float32[:,:,:]))(float32[:, :], int64))
def preproc_obs(infos_array, num_agents):
    observations = np.zeros((infos_array.shape[0], num_agents, 330), dtype = float32)
    share_observations =np.zeros((infos_array.shape[0], num_agents, 220), dtype = float32)
    for idx, info_array in enumerate(infos_array):
        obs, share_obs = thread_processing(
            info = np.ascontiguousarray(info_array),
            num_agents =  num_agents
        )
        observations[idx, : , :] = obs
        share_observations[idx, : , :] = share_obs


    return (observations, share_observations) # (Rollout, num_agents, 330) / (Rollout, num_agents, 220)


def additional_obs(infos, num_agents):
    infos_list = []
    for info in infos:
        info_array = np.zeros(279)
        
        info_array[0  : num_agents] = info["active"]

        num_teammate = num_agents + 1
        num_player = num_teammate * 2
        
        info_array[10 : 10 + num_player] = info["left_team"].reshape(-1)
        info_array[32 : 32 + num_player] = info["left_team_direction"].reshape(-1)
        info_array[54 : 54 + num_teammate] = info["left_team_tired_factor"]
        info_array[65 : 65 + num_teammate] = info["left_team_yellow_card"]
        info_array[76 : 76 + num_teammate] = info["left_team_active"]

        info_array[87  : 87 + num_player] = info["right_team"].reshape(-1)
        info_array[109 : 109 + num_player] = info["right_team_direction"].reshape(-1)
        info_array[131 : 131 + num_teammate] = info["right_team_tired_factor"]
        info_array[142 : 142 + num_teammate] = info["right_team_yellow_card"]
        info_array[153 : 153 + num_teammate] = info["right_team_active"]
        info_array[164 : 164 + 10*num_agents] = info["sticky_actions"].reshape(-1)
        
        info_array[264 : 266] = info["score"]
        info_array[266 : 269] = info["ball"]
        info_array[269 : 272] = info['ball_direction']
        info_array[272 : 275] = info['ball_rotation']

        info_array[275] = info["ball_owned_team"]
        info_array[276] = info["game_mode"]
        info_array[277] = info["steps_left"]
        info_array[278] = info["ball_owned_player"]
        
        if num_agents < 10:
            info_array[num_agents : 10] = -1
            info_array[10 + num_player : 32] = -1
            info_array[32 + num_player : 54] = -1
            info_array[54 + num_teammate : 65] = -1
            info_array[65 + num_teammate : 76] = -1
            info_array[76 + num_teammate : 87] = -1
            info_array[87 + num_player : 109] = -1
            info_array[109 + num_player : 131] = -1
            info_array[131 + num_teammate : 142] = -1
            info_array[142 + num_teammate : 153] = -1
            info_array[153 + num_teammate : 164] = -1
            info_array[164 + 10*num_agents: 264] = -1
            
            
        infos_list.append(info_array)
    infos_array = np.array(infos_list, dtype=np.float32)
    obs , share_obs = preproc_obs(infos_array, num_agents)
    return obs , share_obs

@numba.njit(Tuple((float32[:,:,:], float32[:,:,:]))(float32[:, :, :]))
def init_obs(obs):
    rollout = obs.shape[0]
    agents = obs.shape[1]
    
    init_obs = np.zeros((rollout, agents, 330), dtype=float32)
    init_share_obs = np.zeros((rollout, agents, 220), dtype=float32)
    
    left_position = obs[:, : , 0 : 22]
    left_direction = obs[:, :, 22 : 44]
    right_position = obs[:, : , 44 : 66]
    right_direction = obs[:, : , 66 : 88]
    ball_position = obs[:, :, 88 : 91]
    ball_direction = obs[ : , : , 91 : 94]
    ball_ownership = obs[ : , : , 94 : 97]
    game_mode = obs[:, : , 108 : 115]

    for roll_id in range(rollout):
        for agent_id in range(agents):
            
            active_position = np.ascontiguousarray(left_position[roll_id, agent_id, 2*(agent_id + 1): 2*(agent_id + 1) + 2])
            active_direction = np.ascontiguousarray(left_direction[roll_id, agent_id, 2*(agent_id + 1): 2*(agent_id + 1) + 2])
            relative_ball_position = ball_position[roll_id, agent_id,:2] - active_position
            distance2ball = np.linalg.norm(relative_ball_position)
            relative_ball_owner_position = np.zeros(2, dtype = float32) - active_position
            contigous_left_position = np.ascontiguousarray(left_position[roll_id, agent_id, :])
            contigous_right_position = np.ascontiguousarray(right_position[roll_id, agent_id, :])
            relative_left_position = np.ascontiguousarray(contigous_left_position.reshape(2, 11) - active_position.reshape(2, 1))
            distance2left = np.linalg.norm(relative_left_position, 1)
            relative_left_position = relative_left_position.reshape(-1)
            relative_right_position = np.ascontiguousarray(contigous_right_position.reshape(2, 11) - active_position.reshape(2, 1))
            distance2right = np.linalg.norm(relative_right_position, 1)
            relative_right_position = relative_right_position.reshape(-1)
            
            active_info = np.zeros(87)
            active_info[0: 10] = 0 # sticky actions
            active_info[10: 12] = active_position # active position
            active_info[12: 14] = active_direction # active_direction
            active_info[14] = 0
            active_info[15] = 0
            active_info[16] = 0
            active_info[17] = 0
            active_info[18 : 20] = relative_ball_position 
            active_info[20] = distance2ball 
            active_info[21 : 43] = relative_left_position
            active_info[43: 54] = distance2left
            active_info[54: 76] = relative_right_position
            active_info[76: 87] = distance2right            
            
            ball_own_active_info = np.zeros(57)
            ball_own_active_info[0:3] = ball_position[roll_id, agent_id]
            ball_own_active_info[3:6] = ball_direction[roll_id, agent_id]
            ball_own_active_info[6:9] = ball_ownership[roll_id, agent_id]
            ball_own_active_info[9:12] = 0 # ball rotation
            ball_own_active_info[34] = 1 # ball_owned_player 
            ball_own_active_info[35:37] = active_position
            ball_own_active_info[37:39] = active_direction
            ball_own_active_info[39] = 0   # active_tired_factor
            ball_own_active_info[40] = 0   # active_yellow_card
            ball_own_active_info[41] = 0   # active_red_card
            ball_own_active_info[42] = 0   # active_offside            
            ball_own_active_info[43: 45] = relative_ball_position            
            ball_own_active_info[45] = distance2ball
            ball_own_active_info[46: 48] = 0 # ball_owned_player_pos
            ball_own_active_info[48: 50] = 0 #new_ball_owned_player_direction
            ball_own_active_info[50: 52] = relative_ball_owner_position  # relative_ball_owner_position
            ball_own_active_info[52] = np.linalg.norm(relative_ball_owner_position)
            ball_own_active_info[53: 57] = 0 # ball_owner_info
            
            left_team = np.zeros(88)
            left_team[0 : 22] = left_position[roll_id, agent_id]
            left_team[22 : 44] = left_direction[roll_id, agent_id]
            left_team[44 : 88] = 0      
            
            right_team = np.zeros(88)
            right_team[0 : 22] = right_position[roll_id, agent_id]
            right_team[22 : 44] = right_direction[roll_id, agent_id]
            right_team[44 : 88] = 0     
            
            match_state = np.zeros(9)
            match_state[0:7] = game_mode[roll_id, agent_id]
            match_state[7] = 0 # goal_diff_ratio
            match_state[8] = 1 # steps_left_ratio       
            
            init_obs[roll_id, agent_id, 0] = agent_id + 1
            init_obs[roll_id, agent_id, 1:88] = active_info
            init_obs[roll_id, agent_id, 88:145] = ball_own_active_info
            init_obs[roll_id, agent_id, 145:233] = left_team
            init_obs[roll_id, agent_id, 233:321] = right_team
            init_obs[roll_id, agent_id, 321:330] = match_state
        
    init_share_obs[:,:,0 : 3] = ball_position
    init_share_obs[:,:,3 : 6] = 0
    init_share_obs[:,:,6 : 9] = ball_ownership
    init_share_obs[:,:,9 : 12] = 0
    init_share_obs[:,:,34] = 1
    init_share_obs[:,:,35: 123] = left_team
    init_share_obs[:,:,123: 211] = right_team     
    init_share_obs[:,:,211: 220] = match_state  

    return (init_obs , init_share_obs)
