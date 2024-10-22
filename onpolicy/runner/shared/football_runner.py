from collections import defaultdict
import os
import json
import time

import numpy as np
import torch
import wandb

from utils.util import update_linear_schedule
from runner.shared.base_runner import Runner


from algorithms.utils.obs_preprocessing_final import init_obs, preprocessing



def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        self.difficulty_level = 1
        self.cumulative_win_rate = 0
        self.level_stack = 0
        np.set_printoptions(threshold=np.inf)
        """
        scenario 파일에는 init_level 파일 경로를 넣어야 한다. 
        """
        self.level_file_path = self.all_args.level_dir
        if os.path.exists(self.level_file_path):
            os.remove(self.level_file_path)
        
    def run(self):

        total_num_steps = 0
        eval_stack = 0
        
        while self.num_env_steps >= total_num_steps:
            start_time = time.time()
            
            self.warmup()  

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(total_num_steps, self.num_env_steps)    
                
            done_rollouts = [None for _ in range(self.n_rollout_threads)]
            infos_rollouts = [None for _ in range(self.n_rollout_threads)]

            step = 0
            
            while (step < self.episode_length) and (None in done_rollouts):
                # Sample actions
                past_share_obs = self.buffer.share_obs[step]
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                print(np.array(actions_env).shape)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # rewards =(rewards / self.num_agents) * 10
                obs , share_obs, added_rewards = preprocessing(
                    infos = infos, 
                    obs = obs,
                    past_share_obs = past_share_obs, 
                    actions_env = actions_env,
                    num_agents = self.num_agents, 
                    episode_length = self.episode_length
                )
                
                rewards += added_rewards
                infos = list(infos)
                
                for idx, done in enumerate(dones):
                    if (True in done) and (done_rollouts[idx] == None):
                        done_rollouts[idx] = step + 1
                        info = infos[idx]
                        infos_rollouts[idx] = info
                        # if (info["score"][0] == info["score"][1]):
                        #     rewards[idx] -= 0.5
    
                    if done_rollouts[idx] != None:
                        dones[idx] = [True for _ in range(self.num_agents)]        
                        infos[idx] = infos_rollouts[idx]
                
                infos = tuple(infos)

                data = step, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                # insert data into buffer
                self.insert(data)
                
                step += 1
                
            done_rollouts = [self.episode_length if done_rollout == None else done_rollout for done_rollout in done_rollouts]
            
            total_num_steps += int(np.average(done_rollouts))
            eval_stack += int(np.average(done_rollouts))
            
            # for step_idx, roll_idx in enumerate(self.buffer.rewards):
            #     print(step_idx, roll_idx[0][0], roll_idx[1][0])
            for roll_idx, done_step in enumerate(done_rollouts):
                if done_step < self.episode_length:
                    self.buffer.share_obs[done_step+1: , roll_idx, :] = 0
                    self.buffer.obs[done_step+1: , roll_idx, :] = 0
                    self.buffer.rewards[done_step: , roll_idx, :] = 0
                    self.buffer.actions[done_step: , roll_idx, :] = 0
                    self.buffer.action_log_probs[done_step: , roll_idx, :] = 0
                    self.buffer.masks[done_step+1: , roll_idx, :] = 0
            
            rewards_record = self.buffer.rewards 
            self.compute()
            train_infos = self.train()
            
            train_infos["episode_length"] = np.average(done_rollouts)
            train_infos["total_episode_rewards"] = np.sum(rewards_record) / self.num_agents

            end_time = time.time()
            
            train_infos["Episode_Time"] = end_time - start_time
            
            print(f"\nEnv {self.env_name} Algo {self.algorithm_name} Exp {self.experiment_name} updates {total_num_steps}/{self.num_env_steps} steps in {(end_time - start_time):.2f}")  
            print("total episode rewards is {}".format(train_infos["total_episode_rewards"]))


            train_infos["difficulty_level"] = self.difficulty_level
            # train_infos["cumulative_eval_win_rate"] = self.cumulative_win_rate
            train_infos["level_stack"] = self.level_stack

            self.supervisor(
                win_rate = self.buffer.env_infos["train_win_rate"],
                num_agents = self.num_agents,
                total_num_steps = total_num_steps,
            )
            
            self.log_train(train_infos, total_num_steps)
            self.log_env(self.buffer.env_infos, total_num_steps)
            self.buffer.env_infos = defaultdict(list)
            
            if eval_stack >= self.eval_interval:
               self.eval(total_num_steps)
               eval_stack = 0




    def warmup(self):
        # reset env
        self.buffer.step = 0
        default_obs = self.envs.reset()

        obs , share_obs  = init_obs(
            obs = default_obs, 
            num_agents = self.num_agents, 
            episode_length = self.episode_length
        )

        self.buffer.share_obs[0] = share_obs[:, :self.num_agents, :]
        self.buffer.obs[0] = obs[:, :self.num_agents, :]

    def supervisor(self, win_rate, num_agents, total_num_steps):
        # self.cumulative_win_rate =  self.decay_rate * self.cumulative_win_rate + (1-self.decay_rate) * np.mean(win_rate)
        # if self.cumulative_win_rate > 0.5:
        if np.mean(win_rate) >= 0.8:
            self.level_stack += 1
            if self.level_stack >= 10:
                self.difficulty_level += 1
                # self.cumulative_win_rate = 0
                self.level_stack = 0
                if self.save_model:
                    self.save(total_num_steps = total_num_steps, difficulty_level = self.difficulty_level) 
        result = {
            "difficulty_level": self.difficulty_level,
            "num_agents": num_agents,
        }
        with open(self.level_file_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic= self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )
        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads)) # 10, 3, 1 
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]
        
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        step, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)
        
        if np.all(dones_env) or step == (self.episode_length - 1):
            for idx, info in enumerate(infos):
                goal_diff = info["score"][0] - info["score"][1]   
                if goal_diff > 0:
                    self.buffer.env_infos["train_win_rate"].append(1)
                    self.buffer.env_infos["train_draw_rate"].append(0)
                    self.buffer.env_infos["train_lose_rate"].append(0)
                    
                elif goal_diff == 0:
                    self.buffer.env_infos["train_win_rate"].append(0)
                    self.buffer.env_infos["train_draw_rate"].append(1)
                    self.buffer.env_infos["train_lose_rate"].append(0)

                else:
                    self.buffer.env_infos["train_win_rate"].append(0)
                    self.buffer.env_infos["train_draw_rate"].append(0)
                    self.buffer.env_infos["train_lose_rate"].append(1)

        # reset rnn and mask args for done envs
        
        
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        

            
        self.buffer.insert(
            share_obs=share_obs[:,:self.num_agents, :],
            obs=obs[:,:self.num_agents],
            rnn_states_actor=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks,
        ) 

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_obs , eval_share_obs = init_obs(obs = eval_obs, num_agents = self.num_agents, episode_length = self.episode_length)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_win_rate = np.zeros(self.eval_episode)
        eval_draw_rate = np.zeros(self.eval_episode)
        eval_lose_rate = np.zeros(self.eval_episode)

        step = 0
        quo = self.eval_episode // self.n_eval_rollout_threads
        rem = self.eval_episode % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        
        rollout_ranks = []
        # loop until enough episodes
        while num_done < self.eval_episode or step < self.episode_length:
            eval_obs = eval_obs
            past_share_obs = eval_share_obs
            
            # get actions
            self.trainer.prep_rollout()

            if self.algorithm_name == "tizero":
            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
                eval_actions, eval_rnn_states = self.trainer.policy.act(
                    obs = np.concatenate(eval_obs),
                    rnn_states_actor = np.concatenate(eval_rnn_states),
                    masks = np.concatenate(eval_masks),
                    deterministic =self.all_args.eval_deterministic,
                )
            elif self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = self.trainer.policy.act(
                    cent_obs = np.concatenate(eval_share_obs),
                    obs = np.concatenate(eval_obs),
                    rnn_states_actor = np.concatenate(eval_rnn_states),
                    masks = np.concatenate(eval_masks),
                    deterministic=self.all_args.eval_deterministic,
                )
            elif  self.algorithm_name == "newmodel":
                eval_actions, eval_rnn_states = self.trainer.policy.act(
                    obs = np.concatenate(eval_obs),
                    rnn_states_actor = np.concatenate(eval_rnn_states),
                    masks = np.concatenate(eval_masks),
                    deterministic=self.all_args.eval_deterministic,
                )        
            else:
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
            eval_obs, _, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            scores = [info["score"] for info in eval_infos]
            
            eval_obs , eval_share_obs, _ = preprocessing(
                infos = eval_infos, 
                obs = eval_obs,
                past_share_obs = past_share_obs, 
                actions_env = eval_actions_env,
                num_agents = self.num_agents, 
                episode_length = self.episode_length
            )

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goal_diff = scores[idx_env][0] - scores[idx_env][1]
                        
                        if eval_goal_diff > 0:
                            eval_win_rate[num_done] = 1
                            eval_draw_rate[num_done] = 0
                            eval_lose_rate[num_done] = 0
                            rollout_ranks.append([0, 1])
                            
                        elif eval_goal_diff == 0:
                            eval_win_rate[num_done] = 0
                            eval_draw_rate[num_done] = 1
                            eval_lose_rate[num_done] = 0
                            rollout_ranks.append([0, 0])
                        else:
                            eval_win_rate[num_done] = 0
                            eval_draw_rate[num_done] = 0
                            eval_lose_rate[num_done] = 1
                            rollout_ranks.append([1, 0])
                            
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            step += 1

        # get expected goal
        eval_win_rate = np.mean(eval_win_rate)
        eval_draw_rate = np.mean(eval_draw_rate)
        eval_lose_rate = np.mean(eval_lose_rate)
        
        # self.supervisor(
        #     win_rate = self.buffer.env_infos["train_win_rate"],
        #     num_agents = self.num_agents,
        #     total_num_steps = total_num_steps,
        # )
        
        # log and print
        print(f"| eval_win_rate {eval_win_rate} | eval_draw_rate {eval_draw_rate} | eval_lose_rate {eval_lose_rate} | ")
        if self.use_wandb:
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_draw_rate": eval_draw_rate}, step=total_num_steps)
            wandb.log({"eval_lose_rate": eval_lose_rate}, step=total_num_steps)
            
        

