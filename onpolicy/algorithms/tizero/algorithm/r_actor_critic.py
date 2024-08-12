import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer

from algorithms.utils.popart import PopArt

from utils.util import get_shape_from_obs_space

from algorithms.utils.input_encoder import get_fc
from algorithms.utils.input_encoder import FcEncoder
from algorithms.utils.input_encoder import ACTLayer
from algorithms.utils.input_encoder import ObsEncoder, InputEncoder, InputEncoder_critic


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()

        self.device = device
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        self.obs_encoder = ObsEncoder(
            InputEncoder(),
            input_embedding_size=265,
            hidden_size=self.hidden_size,
            _recurrent_N=1,
            _use_orthogonal=True,
            device=self.device,
        )
        self.action_dim = 19
        self.active_id_size = 1
        self.id_max = 11

        self.predict_id = get_fc(self.hidden_size + self.action_dim, self.id_max)
        self.id_embedding = get_fc(self.id_max, self.id_max)

        self.before_act_wrapper = FcEncoder(2, self.hidden_size + self.id_max, self.hidden_size)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(self.device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        active_id = obs[:, : self.active_id_size].squeeze(1).long().to(self.device)
        id_onehot = torch.eye(self.id_max).to(self.device)[active_id]
        id_output = self.id_embedding(id_onehot)

        obs = obs[:, self.active_id_size :]
        obs_output, rnn_states = self.obs_encoder(obs, rnn_states, masks)

        output = torch.cat([id_output, obs_output], 1)

        output = self.before_act_wrapper(output)

        actions, action_log_probs = self.act(output, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        id_groundtruth = obs[:, : self.active_id_size].squeeze(1).long()
        id_onehot = torch.eye(self.id_max).to(self.device)[id_groundtruth]
        obs = obs[:, self.active_id_size :]

        obs_output, rnn_states = self.obs_encoder(obs, rnn_states, masks)
        id_output = self.id_embedding(id_onehot)

        action_onehot = torch.eye(self.action_dim).to(self.device)[action.squeeze(1).long()]
        id_prediction = self.predict_id(torch.cat([obs_output, action_onehot], 1))
        output = torch.cat([id_output, obs_output], 1)

        output = self.before_act_wrapper(output)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            output, action, available_actions, active_masks=active_masks if self._use_policy_active_masks else None
        )

        return action_log_probs, dist_entropy, id_prediction, id_groundtruth


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()

        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.obs_encoder = ObsEncoder(
            InputEncoder_critic(),
            input_embedding_size=128,
            hidden_size=self.hidden_size,
            _recurrent_N=1,
            _use_orthogonal=True,
            device=device,
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        x, rnn_states = self.obs_encoder(cent_obs, rnn_states, masks)

        values = self.v_out(x)

        return values, rnn_states
