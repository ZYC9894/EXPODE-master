import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl


class PredictAtten(nn.Module):
    def __init__(self, obs_shape, args):
        super(PredictAtten, self).__init__()
        self.name = 'predict_atten'
        self.args = args
        self.n_agents = args.n_agents
        self.obs_dim = obs_shape
        self.n_actions = args.n_actions
        self.sa_dim = self.obs_dim + self.n_actions
        self.n_head = args.n_head  # attention head num

        self.embed_dim = args.atten_embed_dim

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        if getattr(args, "hypernet_layers", 1) == 1:
            for i in range(self.n_head):  # multi-head attention
                self.selector_extractors.append(nn.Linear(self.obs_dim, self.embed_dim, bias=False))  # query
                self.key_extractors.append(nn.Linear(self.sa_dim, self.embed_dim, bias=False))
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            for i in range(self.n_head):  # multi-head attention
                selector_nn = nn.Sequential(nn.Linear(self.obs_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim, bias=False))
                self.selector_extractors.append(selector_nn)  # query
                
                key_extractors_nn = nn.Sequential(nn.Linear(self.sa_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim, bias=False))
                self.key_extractors.append(key_extractors_nn)
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 embednet layers is not implemented!")
        else:
            raise Exception("Error setting number of embednet layers.")

    def forward(self, batch, t, ret_attn_logits="mean"):
        obs, actions = self.build_input(batch, t)
        obs_actions = th.cat([obs, actions], dim=-1).permute(1, 0, 2)
        # obs = obs.permute(1, 0, 2)

        # states: (batch_size, agent_num, obs_dim)

        all_head_selectors = [sel_ext(obs) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, agent_num, embed_dim)
        # obs_actions: (agent_num, batch_size, sa_dim)
        all_head_keys = [[k_ext(enc) for enc in obs_actions] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, agent_num, embed_dim)

            # (batch_size, agent_num, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector, th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, agent_num, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)

            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, agent_num, agent_num)
            # (batch_size, 1, agent_num) * (batch_size, 1, agent_num)
            
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        # regularize magnitude of attention logits
        # attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        # head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(dim=1).sum(1).mean()) for probs in head_attend_weights]

        head_attend_logits = th.stack(head_attend_logits, dim=-2) / np.sqrt(self.embed_dim)
        if ret_attn_logits == 'max':
            attn_logits_weight = F.softmax(head_attend_logits.max(dim=-2)[0], dim=-1)
        elif ret_attn_logits == 'mean':
            attn_logits_weight = F.softmax(head_attend_logits.mean(dim=-2), dim=-1)
            
        return head_attend_logits.reshape(batch.batch_size, -1, self.args.n_agents, self.args.n_agents)[:,1:], attn_logits_weight.reshape(batch.batch_size, -1, self.args.n_agents, self.args.n_agents)[:,1:]

    def build_input(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, :t])  # bTav
        if self.args.obs_last_action:
            last_actions = th.zeros_like(batch["actions_onehot"][:, :t])
            last_actions[:, 1:] = batch["actions_onehot"][:, :t-1]
            inputs.append(last_actions)
            # print('obs_last_action')
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t, -1, -1))
            # print('obs_agent_id')

        inputs = th.cat([x.reshape(bs*t, self.n_agents, -1) for x in inputs], dim=2)
        return inputs, batch["actions_onehot"][:, :t].reshape(bs*t, self.n_agents, -1)