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

import cProfile, io, pstats
from pstats import SortKey

from utils.util import update_linear_schedule
from runner.shared.base_runner import Runner


from algorithms.utils.obs_preprocessing import additional_obs, init_obs



def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)

    def run(self):

        total_num_steps = 0
        interval_stack = 0
        
        while self.num_env_steps >= total_num_steps:
            start_time = time.time()
            # pr = cProfile.Profile()
            # pr.enable()
            self.warmup()  
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(total_num_steps, self.num_env_steps)    
                
            done_rollouts = [None for _ in range(self.n_rollout_threads)]
            infos_rollouts = [None for _ in range(self.n_rollout_threads)]

            step = 0
            
            while (step < 3000) and (None in done_rollouts):
                
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                share_obs = obs
                
                if self.use_xt:
                    rewards = self.cal_xt.controller(
                        step = step,
                        rewards = rewards,
                        obs = obs,
                        score = np.array([info["score"] for info in infos]),
                    )
                
                if self.use_additional_obs:
                    obs, share_obs = additional_obs(infos = infos)
                    
                for idx, done in enumerate(dones):
                    if (True in done) and (done_rollouts[idx] == None):
                        done_rollouts[idx] = step + 1
                        infos_rollouts[idx] = infos[idx]
                    if done_rollouts[idx] != None:
                        dones[idx] = [True for _ in range(self.num_agents)]
                        infos = list(infos)
                        infos[idx] = infos_rollouts[idx]
                        infos = tuple(infos)

                data = step, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                # insert data into buffer
                self.insert(data)
                
                step += 1

            done_steps = [3000 if done_rollout == None else done_rollout for done_rollout in done_rollouts]
            
            self.buffer.step = 0
            total_num_steps += int(np.average(done_steps))
            interval_stack += int(np.average(done_steps))
            
            for roll_idx, done_step in enumerate(done_steps):
                if done_step < 3000:
                    self.buffer.share_obs[done_step+1: , roll_idx, :] = 0
                    self.buffer.obs[done_step+1: , roll_idx, :] = 0
                    self.buffer.rewards[done_step: , roll_idx, :] = 0
                    self.buffer.actions[done_step: , roll_idx, :] = 0
                    self.buffer.action_log_probs[done_step: , roll_idx, :] = 0
                    self.buffer.masks[done_step+1: , roll_idx, :] = 0
                    
            # for idxes, step in enumerate(self.buffer.rewards):
            #     for idx, rew in enumerate(step):
            #         if np.any(rew != 0.0):
            #             print(f"step: {idxes} rollout: {idx} {rew[0]}")
            
                    
            rewards_record = self.buffer.rewards # 리셋 대비용 백업
            
            self.compute()
            train_infos = self.train()
            
            # save model
            # if interval_stack >= self.save_interval:
            #     self.save()
            #     interval_stack = 0
            
            
            train_infos["episode_length"] = np.average(done_steps)
            train_infos["total_episode_rewards"] = np.sum(rewards_record) / self.num_agents

            train_infos["Difficulty_level"] =  self.director.level
            end_time = time.time()
            train_infos["Episode_Time"] = end_time - start_time
            
            print(f"\nEnv {self.env_name} Algo {self.algorithm_name} Exp {self.experiment_name} updates {total_num_steps}/{self.num_env_steps} steps in {(end_time - start_time):.2f}")  
            print("total episode rewards is {}".format(train_infos["total_episode_rewards"]))

            self.log_train(train_infos, total_num_steps)
            self.director.assessing_game(goal_diffs = self.buffer.env_infos["train_goal_diff"])
            
            self.log_env(self.buffer.env_infos, total_num_steps)
            self.buffer.env_infos = defaultdict(list)

            # pr.disable()
            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())

            if interval_stack >= self.eval_interval:
                self.eval(total_num_steps)
                interval_stack = 0

    def warmup(self):
        # reset env
        default_obs = self.envs.reset()
        if self.use_additional_obs:
            obs, share_obs = init_obs(obs = default_obs)
            self.buffer.share_obs[0] = share_obs
            self.buffer.obs[0] = obs
            self.buffer.rnn_states[0] = 0
            self.buffer.rnn_states_critic[0] = 0
            self.buffer.masks[0] = 1
        else:
            self.buffer.share_obs[0] = default_obs.copy()
            self.buffer.obs[0] = default_obs.copy()
            self.buffer.rnn_states[0] = 0
            self.buffer.rnn_states_critic[0] = 0
            self.buffer.masks[0] = 1

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
        step, obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)
        
        if np.all(dones_env) or step == 2999:
            for _, info in zip(dones_env, infos):
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
        if self.use_additional_obs:
            eval_obs, _ = init_obs(np.ascontiguousarray(eval_obs))
        
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
                observations, _ = additional_obs(infos = eval_infos)
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


    
