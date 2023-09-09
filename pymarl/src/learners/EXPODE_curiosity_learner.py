import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
from utils.torch_utils import to_cuda
import numpy as np
# from .vdn_Qlearner import vdn_QLearner
from .EXPODE_q_learner import QLearner
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import os


class EXPODE_curiosity_Learner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac_1 = mac
        self.mac_2 = copy.deepcopy(mac)
        self.logger = logger

        self.params_1 = list(self.mac_1.parameters())
        self.params_2 = list(self.mac_2.parameters())

        self.last_target_update_episode = 0
        self.save_buffer_cnt = 0
        if self.args.save_buffer:
            self.args.save_buffer_path = os.path.join(self.args.save_buffer_path, str(self.args.seed))

        self.mixer = None
        self.qmix_learner = QLearner(mac, scheme, logger, args)
        if args.mixer is not None:
            if args.mixer == 'vdn':
                self.mixer_1 = VDNMixer()
                self.mixer_2 = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer_1 = QMixer(args)
                self.mixer_2 = QMixer(args)
            elif args.mixer == "dmaq":
                self.mixer_1 = DMAQer(args)
                self.mixer_2 = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer_1 = DMAQ_QattenMixer(args)
                self.mixer_2 = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params_1 += list(self.mixer_1.parameters())
            self.params_2 += list(self.mixer_2.parameters())
            self.target_mixer_1 = copy.deepcopy(self.mixer_1)
            self.target_mixer_2 = copy.deepcopy(self.mixer_2)

        self.optimiser_1 = RMSprop(params=self.params_1, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_2 = RMSprop(params=self.params_2, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac_1 = copy.deepcopy(self.mac_1)
        self.target_mac_2 = copy.deepcopy(self.mac_2)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.save_buffer_cnt = 0

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, intrinsic_rewards,
                  show_demo=False, save_data=None, show_v=False, save_buffer=False,ec_buffer=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        self.mac_1.init_hidden(batch.batch_size)
        mac_out_1 = self.mac_1.forward(batch, batch.max_seq_length, batch_inf=True)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        x_mac_out_1 = mac_out_1.clone().detach()
        x_mac_out_1[avail_actions == 0] = -9999999
        max_action_qvals_1, max_action_index_1 = x_mac_out_1[:, :-1].max(dim=3)
        max_action_index_1 = max_action_index_1.detach().unsqueeze(3)
        is_max_action_1 = (max_action_index_1 == actions).int().float()

        self.mac_2.init_hidden(batch.batch_size)
        mac_out_2 = self.mac_2.forward(batch, batch.max_seq_length, batch_inf=True)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        x_mac_out_2 = mac_out_2.clone().detach()
        x_mac_out_2[avail_actions == 0] = -9999999
        max_action_qvals_2, max_action_index_2 = x_mac_out_2[:, :-1].max(dim=3)
        max_action_index_2 = max_action_index_2.detach().unsqueeze(3)
        is_max_action_2 = (max_action_index_2 == actions).int().float()


        if show_demo:
            q_i_data = chosen_action_qvals_1.detach().cpu().numpy()
            q_data = (max_action_qvals_1 - chosen_action_qvals_1).detach().cpu().numpy()

        with th.no_grad():
            # Calculate the Q-Values necessary for the target
            self.target_mac_1.init_hidden(batch.batch_size)
            target_mac_out_1 = self.target_mac_1.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
            self.target_mac_2.init_hidden(batch.batch_size)
            target_mac_out_2 = self.target_mac_2.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

            # Mask out unavailable actions
            target_mac_out_1[avail_actions[:, 1:] == 0] = -9999999
            target_mac_out_2[avail_actions[:, 1:] == 0] = -9999999
            # Max over target Q-Values

            # target_max_qvals_1 = target_mac_out_1.max(dim=3)[0]
            # target_max_qvals_2 = target_mac_out_2.max(dim=3)[0]
            
            # Get actions that maximise live Q (for double q-learning)
            
            mac_out_detach_1 = mac_out_1.clone().detach()
            mac_out_detach_1[avail_actions == 0] = -9999999
            cur_max_actions_1 = mac_out_detach_1[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals_1 = th.gather(target_mac_out_1, 3, cur_max_actions_1).squeeze(3)
            mac_out_detach_2 = mac_out_2.clone().detach()
            mac_out_detach_2[avail_actions == 0] = -9999999
            cur_max_actions_2 = mac_out_detach_2[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals_2 = th.gather(target_mac_out_2, 3, cur_max_actions_2).squeeze(3)
        
        # target_max_qvals = th.min(target_max_qvals_1, target_max_qvals_2)


        # Mix
        if self.mixer_1 is not None and self.mixer_2 is not None:
            if self.args.mixer == 'vdn':
                chosen_action_qvals_1 = self.mixer_1(chosen_action_qvals_1, batch["state"][:, :-1])
                chosen_action_qvals_2 = self.mixer_2(chosen_action_qvals_2, batch["state"][:, :-1])
                with th.no_grad():
                    target_max_qvals_1 = self.target_mixer_1(target_max_qvals_1, batch["state"][:, 1:])
                    target_max_qvals_2 = self.target_mixer_2(target_max_qvals_2, batch["state"][:, 1:])
            elif self.args.mixer == "qmix":
                chosen_action_qvals_1 = self.mixer_1(chosen_action_qvals_1, batch["state"][:, :-1])
                chosen_action_qvals_2 = self.mixer_2(chosen_action_qvals_2, batch["state"][:, :-1])
                with th.no_grad():
                    target_max_qvals_1 = self.target_mixer_1(target_max_qvals_1, batch["state"][:, 1:])
                    target_max_qvals_2 = self.target_mixer_2(target_max_qvals_2, batch["state"][:, 1:])
            elif self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies =  self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals_1, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals_1, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            target_max_qvals = th.min(target_max_qvals_1, target_max_qvals_2)
        else:
            chosen_action_qvals_1 = chosen_action_qvals_1.sum(-1)
            chosen_action_qvals_2 = chosen_action_qvals_2.sum(-1)
            target_max_qvals_1 = target_max_qvals_1.sum(-1)
            target_max_qvals_2 = target_max_qvals_2.sum(-1)
            target_max_qvals = th.min(target_max_qvals_1, target_max_qvals_2)

        
        # Calculate 1-step Q-Learning targets
        if self.args.use_emdqn:
            ec_buffer.update_counter += 1
            qec_input = chosen_action_qvals_1.clone().detach()
            qec_input_new = []
            for i in range(self.args.batch_size):
                qec_tmp = qec_input[i, :]
                for j in range(1, batch.max_seq_length):
                    if not mask[i, j - 1]:
                        continue
                    z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())
                    q = ec_buffer.peek(z, None, modify=False)
                    if q != None:
                        qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1]
                        ec_buffer.qecwatch.append(q)
                        ec_buffer.qec_found += 1
                qec_input_new.append(qec_tmp)
            qec_input_new = th.stack(qec_input_new, dim=0)

            # print("qec_mean:", np.mean(ec_buffer.qecwatch))
            episodic_q_hit_pro = 1.0 * ec_buffer.qec_found / self.args.batch_size / ec_buffer.update_counter / batch.max_seq_length
            # print("qec_fount: %.2f" % episodic_q_hit_pro)


        targets_1 = intrinsic_rewards*self.args.is_intrinsic+rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals_1.detach().cpu().numpy()
            tot_target = targets_1.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return
        # Td-error
        td_error_1 = (chosen_action_qvals_1 - targets_1.detach())

        mask = mask.expand_as(td_error_1)
        if self.args.use_emdqn:
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals_1
            emdqn_masked_td_error = emdqn_td_error * mask
        if show_v:
            mask_elems = mask.sum().item()

            actual_v = rewards.clone().detach()
            for t in reversed(range(rewards.shape[1] - 1)):
                actual_v[:, t] += self.args.gamma * actual_v[:, t + 1]
            self.logger.log_stat("test_actual_return", (actual_v * mask).sum().item() / mask_elems, t_env)

            self.logger.log_stat("test_q_taken_mean", (chosen_action_qvals_1 * mask).sum().item() / mask_elems, t_env)
            return
        # 0-out the targets that came from padded data
        masked_td_error_1 = td_error_1 * mask
        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error_1 ** 2).sum() / mask.sum() + q_attend_regs
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss
        else:
            loss_1 = (masked_td_error_1 ** 2).sum() / mask.sum()
            if self.args.use_emdqn:
                emdqn_loss_1 = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss_1 += emdqn_loss_1
        masked_hit_prob_1 = th.mean(is_max_action_1, dim=2) * mask
        hit_prob_1 = masked_hit_prob_1.sum() / mask.sum()
        # Optimise
        self.optimiser_1.zero_grad()
        loss_1.backward()
        grad_norm_1 = th.nn.utils.clip_grad_norm_(self.params_1, self.args.grad_norm_clip)
        self.optimiser_1.step()



        targets_2 = intrinsic_rewards*self.args.is_intrinsic+rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error_2 = (chosen_action_qvals_2 - targets_2.detach())
        mask = mask.expand_as(td_error_2)
        if self.args.use_emdqn:
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals_2
            emdqn_masked_td_error = emdqn_td_error * mask
        # 0-out the targets that came from padded data
        masked_td_error_2 = td_error_2 * mask
        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error_2 ** 2).sum() / mask.sum() + q_attend_regs
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss
        else:
            loss_2 = (masked_td_error_2 ** 2).sum() / mask.sum()
            if self.args.use_emdqn:
                emdqn_loss_2 = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss_2 += emdqn_loss_2
        masked_hit_prob_2 = th.mean(is_max_action_2, dim=2) * mask
        hit_prob_2 = masked_hit_prob_2.sum() / mask.sum()
        # Optimise
        self.optimiser_2.zero_grad()
        loss_2.backward()
        grad_norm_2 = th.nn.utils.clip_grad_norm_(self.params_2, self.args.grad_norm_clip)
        self.optimiser_2.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss_1.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob_1.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm_1, t_env)
            mask_elems = mask.sum().item()
            if self.args.use_emdqn:
                self.logger.log_stat("e_m Q mean", (qec_input_new * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("em_ Q hit probability", episodic_q_hit_pro, t_env)
                self.logger.log_stat("emdqn_loss", emdqn_loss_1.item(), t_env)
                self.logger.log_stat("emdqn_curr_capacity", ec_buffer.ec_buffer.curr_capacity, t_env)
                self.logger.log_stat("emdqn_weight", self.args.emdqn_loss_weight, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error_1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals_1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets_1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error_1 ** 2, mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False, ec_buffer=None):
        intrinsic_rewards = 0
        if self.args.is_intrinsic:
            intrinsic_rewards = self.qmix_learner.train(batch, t_env, episode_num,save_buffer=False, imac=(self.mac_1, self.mac_2), timac=(self.target_mac_1, self.target_mac_2))
        if self.args.is_prioritized_buffer:
            masked_td_error, mask = self.sub_train(batch, t_env, episode_num, intrinsic_rewards=intrinsic_rewards, show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)
        else:
            self.sub_train(batch, t_env, episode_num ,intrinsic_rewards=intrinsic_rewards, show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)

        # if hasattr(self.args, 'save_buffer') and self.args.save_buffer:
        #     if self.buffer.episodes_in_buffer - self.save_buffer_cnt >= self.args.save_buffer_cycle:
        #         if self.buffer.can_sample(self.args.save_buffer_cycle):
        #             batch_tmp=self.buffer.sample(self.args.save_buffer_cycle, newest=True)
        #             intrinsic_rewards_tmp=self.vdn_learner.train(batch_tmp, t_env, episode_num, save_buffer=True,
        #                                                            imac=self.mac, timac=self.target_mac)
        #             self.sub_train(batch_tmp, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards_tmp,
        #                 show_demo=show_demo, save_data=save_data, show_v=show_v, save_buffer=True)
        #
        #
        #         else:
        #             print('**' * 20, self.buffer.episodes_in_buffer, self.save_buffer_cnt)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(ec_buffer)
         
            self.last_target_update_episode = episode_num
            
        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res

    def _update_targets(self,ec_buffer):
        if self.args.use_emdqn:
            ec_buffer.update_kdtree()
        self.target_mac_1.load_state(self.mac_1)
        self.target_mac_2.load_state(self.mac_2)
        if self.mixer_1 is not None:
            self.target_mixer_1.load_state_dict(self.mixer_1.state_dict())
        if self.mixer_2 is not None:    
            self.target_mixer_2.load_state_dict(self.mixer_2.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.qmix_learner.cuda()
        to_cuda(self.mac_1, self.args.device)
        to_cuda(self.mac_2, self.args.device)
        to_cuda(self.target_mac_1, self.args.device)
        to_cuda(self.target_mac_2, self.args.device)
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

    def load_models(self, path):
        self.mac_1.load_models(path, 1)
        self.mac_2.load_models(path, 2)
        # Not quite right but I don't want to save target networks
        self.target_mac_1.load_models(path, 1)
        self.target_mac_2.load_models(path, 2)
        if self.mixer_1 is not None:
            self.mixer_1.load_state_dict(th.load("{}/mixer_1.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer_1.load_state_dict(th.load("{}/mixer_1.th".format(path), map_location=lambda storage, loc: storage))
        if self.mixer_2 is not None:
            self.mixer_2.load_state_dict(th.load("{}/mixer_2.th".format(path), map_location=lambda storage, loc: storage))    
            self.target_mixer_2.load_state_dict(th.load("{}/mixer_2.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_1.load_state_dict(th.load("{}/opt_1.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser_2.load_state_dict(th.load("{}/opt_2.th".format(path), map_location=lambda storage, loc: storage))
