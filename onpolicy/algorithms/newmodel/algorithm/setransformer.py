import torch
import torch.nn as nn
from algorithms.utils.util import init, check

from algorithms.utils.popart import PopArt

from utils.util import get_shape_from_obs_space

from algorithms.utils.newmodel_utils import *
from algorithms.utils.input_encoder import ACTLayer

class SetTransformer(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(SetTransformer, self).__init__()
        self.device = device
        self.hidden_size = args.hidden_size
        
        self.num_rollouts = args.n_rollout_threads 
        self.num_agents = args.num_agents

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        self.obs_shape = (*obs_shape,)[0]
        self.act_space = action_space.n

        self.attention_dim =128 
        num_head = 4 
        self.num_seed_vector = 2 # 각 rollout 당 value 하나를 보장
        num_encoder_block = 2
        self.rnet_dim = 32
        
        self.embed = nn.Sequential(
            nn.Linear(self.obs_shape, self.attention_dim),
            nn.ReLU(inplace=True)
        )
        
        encoder_blocks = []
        for _ in range(num_encoder_block):
            encoder_blocks.append(
                SetAttentionBlock(
                    d = self.attention_dim,
                    h = num_head,
                    rff = RFF(self.attention_dim),
                )
            )
        self.encoder = nn.ModuleList(encoder_blocks)
        
        # self.encoder_rnet = nn.GRU(input_size=self.attention_dim, hidden_size=self.rnet_dim, batch_first=True)
        
        self.act_layer = ACTLayer(action_space, self.attention_dim, self._use_orthogonal, self._gain)
        
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(
                d = self.attention_dim,
                k = self.num_seed_vector, 
                h = num_head, 
                rff = RFF(self.attention_dim)
            ),
            SetAttentionBlock(
                d = self.attention_dim,
                h = num_head,
                rff = RFF(self.attention_dim),
            )
        )
        
        # self.decoder_rnet = nn.GRU(input_size=self.attention_dim, hidden_size=self.rnet_dim, batch_first=True),
        self.v_net_embedding = self.attention_dim * self.num_seed_vector
        self.v_net = nn.Sequential(
            nn.Linear(self.v_net_embedding, self.v_net_embedding // 2),
            nn.Linear(self.v_net_embedding // 2, 1),
        )
        self.to(self.device)

    def forward(self, obs,  masks, encoder_rnn_states = None, decoder_rnn_states = None, available_actions=None, deterministic=False):
        
        obs = check(obs).to(**self.tpdv)
        if encoder_rnn_states is not None:
            encoder_rnn_states = check(encoder_rnn_states).to(**self.tpdv)
        if decoder_rnn_states is not None:
            decoder_rnn_states = check(decoder_rnn_states).to(**self.tpdv)
             
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        obs = obs.reshape(self.num_rollouts, self.num_agents, self.obs_shape)
        
        x = self.embed(obs)
        
        for block in self.encoder:
            x = block(x)
            
        decoder_input = x
        actions, action_log_probs = self.act_layer(x.reshape(-1, self.attention_dim))
        
        x = self.decoder(decoder_input)

        # x, decoder_rnn_states = self.decoder_rnet(x)
        values = self.v_net(x.reshape(-1, self.v_net_embedding))
        return values, actions, action_log_probs, encoder_rnn_states, decoder_rnn_states

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

        obs = obs.reshape(-1, self.num_agents, self.obs_shape)
        
        x = self.embed(obs)
        
        for block in self.encoder:
            output = block(x)

        action_log_probs, dist_entropy = self.act_layer.evaluate_actions(
            output.reshape(-1, self.attention_dim),
            action, 
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
        )
        x = self.decoder(output)
        values = self.v_net(x.reshape(-1, self.v_net_embedding))

        return values, action_log_probs, dist_entropy

