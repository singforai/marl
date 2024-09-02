import torch
from algorithms.newmodel.algorithm.setransformer import  SetTransformer
from utils.util import update_linear_schedule


class NewModelPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.model = SetTransformer(args, self.obs_space, self.act_space, device = self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, 
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        encoder_rnn_states = rnn_states_actor
        decoder_rnn_states = rnn_states_critic
        
        values, actions, action_log_probs, encoder_rnn_states, decoder_rnn_states = self.model(
            obs,
            masks,
            encoder_rnn_states,
            decoder_rnn_states,
            available_actions,
            deterministic,
            
        )
        rnn_states_actor = encoder_rnn_states
        rnn_states_critic = decoder_rnn_states

        return values, actions, action_log_probs,encoder_rnn_states, decoder_rnn_states

    def get_values(self, obs, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        # encoder_rnn_states = rnn_states_actor
        # decoder_rnn_states = rnn_states_critic
        
        values, _, _, _, _ = self.model(
            obs,
            masks,
        )
        # rnn_states_actor = encoder_rnn_states
        # rnn_states_critic = decoder_rnn_states
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, critic_masks_batch=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if critic_masks_batch is None:
            critic_masks_batch = masks
            
        values, action_log_probs, dist_entropy = self.model.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks
        )
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        
        encoder_rnn_states = rnn_states_actor
        
        _, actions, _, encoder_rnn_states, _ = self.model(
            obs = obs, 
            masks = masks, 
            encoder_rnn_states = encoder_rnn_states,
            deterministic = deterministic
        )

        rnn_states_actor = encoder_rnn_states
        
        return actions, rnn_states_actor


    def save(self,save_dir, policy_actor, time):
        torch.save(policy_actor.state_dict(), str(save_dir) + f"/tizero_actor_{time}.pt")
        

    def restore(self, model_dir, time):
        model_dir = f"{model_dir}/tizero_actor_{str(time)}"
        tizero_actor_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(tizero_actor_state_dict)
        actor = torch.load(str(self.model_dir) + '/actor.pt')
        return actor