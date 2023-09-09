import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.predict_atten import PredictAtten
import torch as th
from torch.optim import RMSprop
from torch.optim import Adam
from utils.torch_utils import to_cuda
from controllers import REGISTRY as mac_REGISTRY
import torch.nn.functional as func
import numpy as np


class QLearner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac_1 = mac
        self.mac_2 = copy.deepcopy(mac)
        self.logger = logger

        self.params_1 = list(self.mac_1.parameters())
        self.params_2 = list(self.mac_2.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer_1 = VDNMixer()
                self.mixer_2 = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer_1 = QMixer(args)
                self.mixer_2 = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params_1 += list(self.mixer_1.parameters())
            self.params_2 += list(self.mixer_2.parameters())
            self.target_mixer_1 = copy.deepcopy(self.mixer_1)
            self.target_mixer_2 = copy.deepcopy(self.mixer_2)
            self.soft_target_mixer_1 = copy.deepcopy(self.mixer_1)
            self.soft_target_mixer_2 = copy.deepcopy(self.mixer_2)

        self.optimiser_1 = RMSprop(params=self.params_1, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_2 = RMSprop(params=self.params_2, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac_1 = copy.deepcopy(self.mac_1)
        self.target_mac_2 = copy.deepcopy(self.mac_2)
        self.soft_target_mac_1 = copy.deepcopy(self.mac_1)
        self.soft_target_mac_2 = copy.deepcopy(self.mac_2)

        self.predict_mac = mac_REGISTRY[args.mac](scheme, groups, args)
        self.predict_mac_params = list(self.predict_mac.parameters())
        self.predictAtten = PredictAtten(mac.input_shape, self.args)
        self.predictAtten_params = list(self.predictAtten.parameters())
        self.predict_params = self.predict_mac_params + self.predictAtten_params
        self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)

        self.decay_stats_t = 0
        self.decay_stats_t_2 = 0
        self.state_shape = scheme["state"]["vshape"]

        self.log_stats_t = -self.args.learner_log_interval - 1

    def subtrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, save_buffer=False, imac=None, timac=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        prediction_mask = mask.repeat(1, 1, self.args.n_agents)
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac_1.init_hidden(batch.batch_size)
        self.mac_2.init_hidden(batch.batch_size)
        self.predict_mac.init_hidden(batch.batch_size)
        self.target_mac_1.init_hidden(batch.batch_size)
        self.target_mac_2.init_hidden(batch.batch_size)
        self.soft_target_mac_1.init_hidden(batch.batch_size)
        self.soft_target_mac_2.init_hidden(batch.batch_size)
        with th.no_grad():
            target_mac_out_1 = self.target_mac_1.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
            target_mac_out_2 = self.target_mac_2.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Pick the Q-Values for the actions taken by each agent
        mac_out_1 = self.mac_1.forward(batch, batch.max_seq_length, batch_inf=True)
        chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        x_mac_out_1 = mac_out_1.clone().detach()
        x_mac_out_1[avail_actions == 0] = -9999999
        max_action_qvals_1, max_action_index_1 = x_mac_out_1[:, :-1].max(dim=3)
        max_action_index_1 = max_action_index_1.detach().unsqueeze(3)
        is_max_action_1 = (max_action_index_1 == actions).int().float()

        mac_out_2 = self.mac_2.forward(batch, batch.max_seq_length, batch_inf=True)
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        x_mac_out_2 = mac_out_2.clone().detach()
        x_mac_out_2[avail_actions == 0] = -9999999
        max_action_qvals_2, max_action_index_2 = x_mac_out_2[:, :-1].max(dim=3)
        max_action_index_2 = max_action_index_2.detach().unsqueeze(3)
        is_max_action_2 = (max_action_index_2 == actions).int().float()

        with th.no_grad():
            mac_out_detach_1 = mac_out_1.clone().detach()
            mac_out_detach_1[avail_actions == 0] = -9999999
            cur_max_actions_1 = mac_out_detach_1[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals_1 = th.gather(target_mac_out_1, 3, cur_max_actions_1).squeeze(3)
            mac_out_detach_2 = mac_out_2.clone().detach()
            mac_out_detach_2[avail_actions == 0] = -9999999
            cur_max_actions_2 = mac_out_detach_2[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals_2 = th.gather(target_mac_out_2, 3, cur_max_actions_2).squeeze(3)

        # Mix
        if self.mixer_1 is not None and self.mixer_2 is not None:
            chosen_action_qvals_1 = self.mixer_1(chosen_action_qvals_1, batch["state"][:, :-1])
            chosen_action_qvals_2 = self.mixer_2(chosen_action_qvals_2, batch["state"][:, :-1])
            with th.no_grad():
                target_max_qvals_1 = self.target_mixer_1(target_max_qvals_1, batch["state"][:, 1:])
                target_max_qvals_2 = self.target_mixer_2(target_max_qvals_2, batch["state"][:, 1:])

        target_max_qvals = th.min(target_max_qvals_1, target_max_qvals_2)
        # Calculate 1-step Q-Learning targets
        targets_1 = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error_1 = (chosen_action_qvals_1 - targets_1.detach())
        td_mask_1 = mask.expand_as(td_error_1)
        # 0-out the targets that came from padded data
        masked_td_error_1 = td_error_1 * td_mask_1
        # Normal L2 loss, take mean over actual data
        loss_1 = (masked_td_error_1 ** 2).sum() / td_mask_1.sum()
        masked_hit_prob_1 = th.mean(is_max_action_1, dim=2) * td_mask_1
        hit_prob_1 = masked_hit_prob_1.sum() / td_mask_1.sum()
        # Optimise
        self.optimiser_1.zero_grad()
        loss_1.backward()
        grad_norm_1 = th.nn.utils.clip_grad_norm_(self.params_1, self.args.grad_norm_clip)
        self.optimiser_1.step()

        # Calculate 1-step Q-Learning targets
        targets_2 = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error_2 = (chosen_action_qvals_2 - targets_2.detach())
        td_mask_2 = mask.expand_as(td_error_2)
        # 0-out the targets that came from padded data
        masked_td_error_2 = td_error_2 * td_mask_2
        # Normal L2 loss, take mean over actual data
        loss_2 = (masked_td_error_2 ** 2).sum() / td_mask_2.sum()
        masked_hit_prob_2 = th.mean(is_max_action_2, dim=2) * td_mask_2
        hit_prob_2 = masked_hit_prob_2.sum() / td_mask_2.sum()
        # Optimise
        self.optimiser_2.zero_grad()
        loss_2.backward()
        grad_norm_2 = th.nn.utils.clip_grad_norm_(self.params_2, self.args.grad_norm_clip)
        self.optimiser_2.step()


        ''' prediction net update '''
        with th.no_grad():
            soft_target_mac_out_1 = self.soft_target_mac_1.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
            soft_target_mac_out_2 = self.soft_target_mac_2.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]  # bs, max_step, agent_n, action_n

            predict_soft_mask = th.gt(target_max_qvals_1, target_max_qvals_2).unsqueeze(-1).repeat(1, 1, soft_target_mac_out_1.shape[-2], soft_target_mac_out_1.shape[-1]).int() # bs, max_step, 1
            soft_target_mac_out = predict_soft_mask*soft_target_mac_out_1 + (1-predict_soft_mask)*soft_target_mac_out_2

        predict_mac_out = self.predict_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
        soft_target_mac_out_next = soft_target_mac_out.clone().detach()
        soft_target_mac_out_next = soft_target_mac_out_next.contiguous().view(-1, self.args.n_actions) * 10
        predict_mac_out = predict_mac_out.contiguous().view(-1, self.args.n_actions)
        
        prediction_error = func.pairwise_distance(predict_mac_out, soft_target_mac_out_next, p=2.0, keepdim=True)
        # print("soft_target_mac_out_1: ", soft_target_mac_out_1.shape)
        # print("soft_target_mac_out_2: ", soft_target_mac_out_2.shape)
        # print("soft_target_mac_out: ", soft_target_mac_out.shape)
        # print("predict_mac_out: ", predict_mac_out.shape)
        # print("prediction_error: ", prediction_error.shape)
        # print("prediction_mask: ", prediction_mask.shape)

        '''modify'''
        # prediction_error = prediction_error.reshape(batch.batch_size, -1, self.args.n_agents) * prediction_mask

        ''' other idea: predictAtten_q_agent -> soft_target_mac_out, and atten_weight.detach in calculate atten_prediction_error'''
        prediction_error = prediction_error.reshape(batch.batch_size, -1, self.args.n_agents)
        atten_logits, atten_weight = self.predictAtten.forward(batch, batch.max_seq_length)
        atten_prediction_error = ((prediction_error.unsqueeze(-2).repeat(1, 1, self.args.n_agents, 1) * atten_weight).sum(-1)) * prediction_mask
        # assert atten_weight == 0

        prediction_error = prediction_error*prediction_mask

        # prediction_error_mean = prediction_error.mean(dim=-1, keepdim=True).detach()
        # prediction_error_std = th.sqrt(((prediction_error - prediction_error_mean) ** 2).mean(dim=-1, keepdim=True)) 
        # prediction_error = (prediction_error + self.args.std_alpha * prediction_error_std) * prediction_mask
        # attention
        '''modify'''

        if hasattr(self.args, 'mask_other_agents') and self.args.mask_other_agents:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.detach()[:, :, 0:1])
        else:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.mean(dim=-1, keepdim=True).detach())

        '''modify'''
        prediction_loss = prediction_error.sum() / prediction_mask.sum()
        atten_prediction_loss = atten_prediction_error.sum() / prediction_mask.sum()   
        prediction_loss += self.args.atten_loss_scale * atten_prediction_loss
        '''modify'''     
        ############################
        if save_buffer:
            return intrinsic_rewards
        self.predict_optimiser.zero_grad()
        prediction_loss.backward()
        predict_grad_norm = th.nn.utils.clip_grad_norm_(self.predict_params, self.args.grad_norm_clip)
        self.predict_optimiser.step()
        ############################

        if self.args.curiosity_decay:
            if t_env - self.decay_stats_t >= self.args.curiosity_decay_cycle:
                if self.args.curiosity_decay_rate <= 1.0:
                    if self.args.curiosity_scale > self.args.curiosity_decay_stop:
                         self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                    else:
                         self.args.curiosity_scale = self.args.curiosity_decay_stop
                else:
                     if self.args.curiosity_scale < self.args.curiosity_decay_stop:
                         self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                     else:
                         self.args.curiosity_scale = self.args.curiosity_decay_stop

                self.decay_stats_t=t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("vdn loss", loss_1.item(), t_env)
            self.logger.log_stat("curiosity_scale", self.args.curiosity_scale, t_env)
            self.logger.log_stat("curiosity_decay_rate", self.args.curiosity_decay_rate, t_env)
            self.logger.log_stat("curiosity_decay_cycle", self.args.curiosity_decay_cycle, t_env)

            self.logger.log_stat("curiosity_decay_stop", self.args.curiosity_decay_stop, t_env)
            self.logger.log_stat("vdn hit_prob", hit_prob_1.item(), t_env)
            self.logger.log_stat("vdn grad_norm", grad_norm_1, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("vdn prediction loss", prediction_loss.item(), t_env)

            self.logger.log_stat("vdn intrinsic rewards", intrinsic_rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("vdn extrinsic rewards", rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("vdn td_error_abs", (masked_td_error_1.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("vdn q_taken_mean", (chosen_action_qvals_1 * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("vdn target_mean", (targets_1 * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            self.log_stats_t = t_env

        return intrinsic_rewards

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, save_buffer=False, imac=None, timac=None):

        intrinsic_rewards = self.subtrain(batch, t_env, episode_num, save_buffer=save_buffer, imac=imac, timac=timac)

        self._smooth_update_predict_targets()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        return intrinsic_rewards

    def _update_targets(self):
        self.target_mac_1.load_state(self.mac_1)
        self.target_mac_2.load_state(self.mac_2)
        if self.mixer_1 is not None and self.mixer_2 is not None:
            self.target_mixer_1.load_state_dict(self.mixer_1.state_dict())
            self.target_mixer_2.load_state_dict(self.mixer_2.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _smooth_update_predict_targets(self):
        self.soft_update(self.soft_target_mac_1, self.mac_1, self.args.soft_update_tau)
        self.soft_update(self.soft_target_mac_2, self.mac_2, self.args.soft_update_tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        to_cuda(self.mac_1, self.args.device)
        to_cuda(self.mac_2, self.args.device)
        to_cuda(self.target_mac_1, self.args.device)
        to_cuda(self.target_mac_2, self.args.device)
        to_cuda(self.soft_target_mac_1, self.args.device)
        to_cuda(self.soft_target_mac_2, self.args.device)
        to_cuda(self.predict_mac, self.args.device)
        to_cuda(self.predictAtten, self.args.device)
        if self.mixer_1 is not None and self.mixer_2 is not None:
            to_cuda(self.mixer_1, self.args.device)
            to_cuda(self.mixer_2, self.args.device)
            to_cuda(self.target_mixer_1, self.args.device)
            to_cuda(self.target_mixer_2, self.args.device)

    def save_models(self, path):
        self.mac_1.save_models(path, 1)
        self.mac_2.save_models(path, 2)
        if self.mixer_1 is not None:
            th.save(self.mixer_1.state_dict(), "{}/mixer_1.th".format(path))
        if self.mixer_2 is not None:
            th.save(self.mixer_2.state_dict(), "{}/mixer_2.th".format(path))
        th.save(self.optimiser_1.state_dict(), "{}/opt_1.th".format(path))
        th.save(self.optimiser_2.state_dict(), "{}/opt_2.th".format(path))
        th.save(self.predictAtten.state_dict(), "{}/predictAtten.th".format(path))

    def load_models(self, path):
        self.mac_1.load_models(path, 1)
        self.mac_2.load_models(path, 2)
        # Not quite right but I don't want to save target networks
        self.target_mac_1.load_models(path, 1)
        self.target_mac_2.load_models(path, 2)
        self.soft_target_mac_1.load_models(path, 1)
        self.soft_target_mac_2.load_models(path, 2)
        if self.mixer_1 is not None:
            self.mixer_1.load_state_dict(th.load("{}/mixer_1.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer_1.load_state_dict(th.load("{}/mixer_1.th".format(path), map_location=lambda storage, loc: storage))
        if self.mixer_2 is not None:
            self.mixer_2.load_state_dict(th.load("{}/mixer_2.th".format(path), map_location=lambda storage, loc: storage))    
            self.target_mixer_2.load_state_dict(th.load("{}/mixer_2.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_1.load_state_dict(th.load("{}/opt_1.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_2.load_state_dict(th.load("{}/opt_2.th".format(path), map_location=lambda storage, loc: storage))
        self.predictAtten.load_state_dict(th.load("{}/predictAtten.th".format(path), map_location=lambda storage, loc: storage))