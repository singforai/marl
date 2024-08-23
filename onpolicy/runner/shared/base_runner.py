import wandb
import os
from gym import spaces
import numpy as np
import torch
from utils.trueskill import TrueSkill
from datetime import datetime

from utils.shared_buffer import SharedReplayBuffer


from copy import deepcopy

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']        
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs'] 

        # parameters
        self.env_name = self.all_args.env_name
        self.num_agents = self.all_args.num_agents
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # 나의 오리지널
        self.render_mode = self.all_args.render_mode
        self.use_xt = self.all_args.use_xt
        self.eval_episode = self.all_args.eval_episodes
        self.save_model = self.all_args.save_model
        
        if self.save_model:
            self.save_dir = str(f"./models/{self.experiment_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        if self.use_wandb:
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]


        # observation_space = self.envs.observation_space[0]
        # share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else observation_space

        # if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
        #     from algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
        #     from algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy

        # elif self.algorithm_name == "tizero":
        #         from algorithms.tizero.tizero import TiZero as TrainAlgo
        #         from algorithms.tizero.algorithm.TiZeroPolicy import TiZeroPolicy as Policy
                
        # else:
        #     from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        #     from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        
        # low = np.full((330,), -np.inf)
        # high = np.full((330,), np.inf)
        # observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # low = np.full((220,), -np.inf)
        # high = np.full((220,), np.inf)
        # share_observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # # policy network
        # if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
        #     self.policy = Policy(
        #         self.all_args, 
        #         observation_space,
        #         share_observation_space, 
        #         self.envs.action_space[0], 
        #         self.num_agents, 
        #         device = self.device
        #     )
        #     self.enem_policy = Policy(
        #         self.all_args, 
        #         observation_space,
        #         share_observation_space, 
        #         self.envs.action_space[0], 
        #         self.num_agents, 
        #         device = self.device
        #     )

        # else:
        #     self.policy = Policy(
        #         self.all_args, 
        #         observation_space, 
        #         share_observation_space, 
        #         self.envs.action_space[0], 
        #         device = self.device
        #     )
        #     self.opponent_policy = Policy(
        #         self.all_args, 
        #         observation_space, 
        #         share_observation_space, 
        #         self.envs.action_space[0], 
        #         device = self.device
        #     )

        # if self.model_dir is not None:
        #     self.restore(self.model_dir)

        # # algorithm
        # if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
        #     self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        # else:
        #     self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # # buffer
        # self.buffer = SharedReplayBuffer(
        #     self.all_args,
        #     self.num_agents,
        #     observation_space,
        #     share_observation_space,
        #     self.envs.action_space[0]
        # )
        
        # self.opponent_buffer = SharedReplayBuffer(
        #     self.all_args,
        #     self.num_agents,
        #     observation_space,
        #     share_observation_space,
        #     self.envs.action_space[0]
        # )
        
        # self.trueskill = TrueSkill(
        #     args = self.all_args,
        #     num_agents = self.num_agents,
        #     init_draw_prob = 0.9,
        #     init_mu = 15, 
        # )
        

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))

        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        self.opponent_buffer.after_update()

        for key, value in train_infos.items():
            if isinstance(value, torch.Tensor):
                train_infos[key] = value.detach().cpu()

        return deepcopy(train_infos)

    def save(self, total_num_steps, difficulty_level):
        """Save policy's actor and critic networks."""
        self.save_directory = f"{self.save_dir}/level_{difficulty_level}"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_directory, total_num_steps)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), self.save_directory + f"/actor_{total_num_steps}.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), self.save_directory + f"/critic_{total_num_steps}.pt")

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
