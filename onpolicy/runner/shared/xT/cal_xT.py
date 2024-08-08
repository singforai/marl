import pandas as pd
import numpy as np

from collections import defaultdict, deque

class xT():
    def __init__(self, args):
        self.args = args
        self.xt_type = self.args.xt_type
        self.n_rollout_threads = self.args.n_rollout_threads

        xT_csv = f'/home/uosai/Desktop/marl/onpolicy/runner/shared/xT/csv/{self.xt_type}.csv'
        self.df = pd.read_csv(xT_csv, header=None, skiprows=1).values
        self.df_shape = np.array([72, 96])

        if self.xt_type == "compound_xt":
            self.start_xt = -0.0036491860948993
        elif self.xt_type == "base_xt":
            self.start_xt = 0.0061309340141949

    

    
    def initialize_xt(self, roll_idx):
        self.xT_deque[roll_idx] = deque(maxlen=2)
        self.xT_deque[roll_idx].append(self.start_xt)

    
    def cal_xthreat(self, obs):
        if self.args.representation == "extracted":
            ball_location = np.nonzero(obs[0][0][:,:, 2])
        elif self.args.representation == "simple115v2":
            ball_location = obs[0][88:90]
        indiced_ball_location = np.floor(((ball_location + 1) / 2) * self.df_shape).astype(int)
        for idx, index in enumerate(indiced_ball_location):
            if 0 <= index < self.df_shape[idx]:
                pass
            else:
                return "out_of_range"

        return self.df[indiced_ball_location[0]][indiced_ball_location[1]]

    
    def controller(self, step, rewards, obs, score):
        if step == 0:
            self.xT_deque = [deque(maxlen=2) for _ in range(self.n_rollout_threads)]
            for dq in self.xT_deque:
                dq.append(self.start_xt)
            self.score = self.score = [[0, 0] for _ in range(self.n_rollout_threads)]

        for roll_idx, past_score in enumerate(self.score):
            if all(x == y for x, y in zip(past_score, score[roll_idx])):
                threat_value = self.cal_xthreat(obs = obs[roll_idx])
                if threat_value == "out_of_range":
                    pass
                else:
                    self.xT_deque[roll_idx].append(threat_value)
                    xT_score = self.xT_deque[roll_idx][1] - self.xT_deque[roll_idx][0]
                    rewards[roll_idx] += xT_score * 5
            else:
                self.initialize_xt(roll_idx = roll_idx)
        self.score = score
        return rewards
    
