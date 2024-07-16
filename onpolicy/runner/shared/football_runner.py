from collections import defaultdict
from itertools import chain
import os
import time
from itertools import starmap
import imageio
import numpy as np
import torch
import wandb
import importlib

# import cProfile, io, pstats
# from pstats import SortKey

from utils.util import update_linear_schedule
from runner.shared.base_runner import Runner

from runner.shared.xT.cal_xT import xT
from algorithms.utils.obs_preprocessing import preproc_obs

from envs.package.gfootball.scenarios.curriculum_learning_11vs11 import Director

def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        self.game_length = 3000

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            # pr = cProfile.Profile()
            # pr.enable()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            start_time = time.time()
            for step in range(self.episode_length):
                
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                scores = self.infos_processing(infos = infos)

                if self.use_xt:
                    rewards = self.cal_xt.controller(
                        step = step,
                        rewards = rewards,
                        obs = obs,
                        score = scores,
                    )

                share_obs = obs
                if self.use_additional_obs:
                    """
                    TiZero 구현을 위한 관측 정보 변경
                    actor(obs): 330
                    critic(share_obs): 220
                    """
                    observations, share_obs = self.dict2array(infos = infos)
                    obs = observations
                    share_obs = share_obs

                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)


            
            # pr.disable()
            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())

            # compute return and update network

            end_time = time.time()

            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            # if total_num_steps % self.log_interval == 0:
            print(f"\nEnv {self.env_name} Algo {self.algorithm_name} Exp {self.experiment_name} updates {episode}/{episodes} episodes total num timesteps {total_num_steps}/{self.num_env_steps}")     
            train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
            train_infos["Episode_Time"] = end_time - start_time
            train_infos["Difficulty_level"] =  self.director.level
            
            print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

            self.log_train(train_infos, total_num_steps)
            self.director.assessing_game(goal_diffs = self.buffer.env_infos["train_goal_diff"])

            self.buffer.env_infos["train_possession_rate"] = np.array(self.buffer.possession_state) / self.game_length
            self.log_env(self.buffer.env_infos, total_num_steps)
            self.buffer.possession_state = [ 0 for _ in range(self.n_rollout_threads)]
            self.buffer.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    
    def infos_processing(self, infos):
        possessions = [info['ball_owned_team'] for info in infos]
        for idx, possession in enumerate(possessions):
            if possession == -1:
                self.buffer.possession_state[idx] += 1
        scores = [info["score"] for info in infos]
        return scores
    
    def dict2array(self, infos):
        infos_list = []
        for info in infos:
            info_array = np.concatenate((
                info["active"],
                info["left_team"].reshape(-1),
                info["left_team_direction"].reshape(-1), 
                info["left_team_tired_factor"],
                info["left_team_yellow_card"],
                info["left_team_active"],
                info["right_team"].reshape(-1),
                info["right_team_direction"].reshape(-1), 
                info["right_team_tired_factor"],
                info["right_team_yellow_card"],
                info["right_team_active"],
                info["sticky_actions"].reshape(-1),
                info["score"],
                info["ball"],
                info['ball_direction'],
                info['ball_rotation'],
                [info["ball_owned_team"], info["game_mode"], info["steps_left"], info["ball_owned_player"]]
            ))
            infos_list.append(info_array)
        infos_array = np.ascontiguousarray(infos_list, dtype=np.float32)

        obs , share_obs = preproc_obs(infos_array)
        return obs , share_obs

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # self.buffer.share_obs[0] = obs.copy()
        # self.buffer.obs[0] = obs.copy()
        self.director = Director()
        if self.use_xt:
            self.cal_xt = xT(args = self.all_args)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)
        
        if np.any(dones_env):
            for done, info in zip(dones_env, infos):
                if done:
                    goal_diff = info["score"][0] - info["score"][1]

                    self.buffer.env_infos["train_goal_diff"].append(goal_diff)
                    
                    self.buffer.env_infos["train_goal"].append(info["score"][0])
                    if goal_diff > 0:
                        self.buffer.env_infos["train_WDL"].append(1)
                    elif goal_diff == 0:
                        self.buffer.env_infos["train_WDL"].append(0)
                    else:
                        self.buffer.env_infos["train_WDL"].append(-1)
                    
        # reset rnn and mask args for done envs
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            rnn_states_actor=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks
        )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_obs = np.random.rand(self.n_eval_rollout_threads, self.num_agents, 330)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.eval_episode)
        eval_WDL = np.zeros(self.eval_episode)
        eval_goal_diff = np.zeros(self.eval_episode)

        step = 0
        quo = self.eval_episode // self.n_eval_rollout_threads
        rem = self.eval_episode % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.eval_episode or step < self.episode_length:
            # get actions
            self.trainer.prep_rollout()

            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=self.all_args.eval_deterministic
            )
            
            # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            scores = [info["score"] for info in eval_infos]
            
            if self.use_additional_obs:
                observations, _ = self.dict2array(infos = eval_infos)
                eval_obs = observations

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goal_diff[num_done] = (scores[idx_env][0] - scores[idx_env][1])
                        eval_goals[num_done] = scores[idx_env][0]
                        if eval_goal_diff[num_done] > 0:
                            eval_WDL[num_done] = 1
                        elif eval_goal_diff[num_done] == 0:
                            eval_WDL[num_done] = 0
                        else:
                            eval_WDL[num_done] = -1
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1


        # get expected goal
        eval_goal = np.mean(eval_goals)
        eval_WDL = np.mean(eval_WDL)
        eval_goal_diff = np.mean(eval_goal_diff)


        # log and print
        print(f"| eval_goal {eval_goal} | eval_goal_diff {eval_goal_diff} | eval_WDL {eval_WDL} | ")
        if self.use_wandb:
            wandb.log({"eval_goal": eval_goal}, step=total_num_steps)
            wandb.log({"eval_WDL": eval_WDL}, step=total_num_steps)
            wandb.log({"eval_goal_diff": eval_goal_diff}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_goal", {"expected_goal": eval_goal}, total_num_steps)
            self.writter.add_scalars("eval_WDL", {"eval_WDL": eval_WDL}, total_num_steps)
            self.writter.add_scalars("eval_goal_diff", {"eval_goal_diff": eval_goal_diff}, total_num_steps)
 



    @torch.no_grad()
    def render(self):        
        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            render_obs = render_env.reset()
            render_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            render_dones = False
            while not np.any(render_dones):
                self.trainer.prep_rollout()
                render_actions, render_rnn_states = self.trainer.policy.act(
                    np.concatenate(render_obs),
                    np.concatenate(render_rnn_states),
                    np.concatenate(render_masks),
                    deterministic=True
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                render_actions = np.array(np.split(_t2n(render_actions), self.n_rollout_threads))
                render_rnn_states = np.array(np.split(_t2n(render_rnn_states), self.n_rollout_threads))

                render_actions_env = [render_actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

                # step
                render_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)

                # append frame
                if self.all_args.save_gifs:        
                    image = render_infos[0]["frame"]
                    frames.append(image)
            
            # print goal
            render_goals[i_episode] = render_rewards[0, 0]
            print("goal in episode {}: {}".format(i_episode, render_rewards[0, 0]))

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
        
        print("expected goal: {}".format(np.mean(render_goals)))
