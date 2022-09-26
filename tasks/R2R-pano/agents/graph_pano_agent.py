import json
import random
import numpy as np
import copy

from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.autograd import Variable

import networkx as nx

from smna_models.utils import try_cuda
from utils import padding_idx, end_token_idx, pad_list_tensors, kl_div, CrossEntropy, Entropy, SoftBCELoss
from agents.pano_agent import PanoBaseAgent
from models.modules import create_mask

import pdb

class GraphPanoSeq2SeqAgent(PanoBaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """
    def __init__(self, opts, env, results_path, encoder, model, feedback='sample', episode_len=20, actor_network=None):
        super(GraphPanoSeq2SeqAgent, self).__init__(env, results_path)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts    = opts
        self.encoder = encoder
        self.model   = model
        self.actor_network = actor_network
        self.feedback      = feedback
        self.episode_len   = episode_len
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ignore_index  = opts.max_navigable + 1  # we define (max_navigable+1) as ignore since 15(navigable) + 1(STOP)
        self.criterion     = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.cross_entropy = CrossEntropy()
        
        self.cur_iteration = 0


    def init_traj(self, obs):
        """initialize the trajectory"""
        batch_size = len(obs)

        traj, scan_id = [], []
        for ob in obs:
            traj.append(OrderedDict([
                ('instr_id', ob['instr_id']),
                ('path_id', ob['path_id'] if 'path_id' in ob.keys() else None),
                ('path', [(ob['viewpoint'], ob['heading'], ob['elevation'])]),
                ('length', 0),
                ('teacher', ob['teacher'] if 'teacher' in ob.keys() else None),
                ('scan', ob['scan']),
                ('feature', [ob['feature']]),
                ('action_u', [torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)]),
                ('adj_loc_list', [ob['adj_loc_list']]),
                ('action_embedding', [ob['action_embedding']]),
            ]))
            scan_id.append(ob['scan'])

        self.traj_length = [1] * batch_size
        self.value_loss = torch.tensor(0).float().to(self.device)

        ended = np.array([False] * batch_size)
        last_recorded = np.array([False] * batch_size)

        return traj, scan_id, ended, last_recorded


    def init_GSA_traj(self, batch_size):
        GSA_traj = []
        for i in range(batch_size):
            GSA_traj.append(OrderedDict([
                ('policy', []),
                ('action', []),
                ('logit', []),
                ('graph_teacher', []),
                ('batch_q_values_0', []),
                ('batch_q_values_1', []),
                ('batch_queue_path', []),
                ('select_queue_path', []),
                ('current_traj_path', []),
                ('traj_mark', []),
                ('ended', []),
                ('reward', []),
                ('pooling_att', []),
            ]))

        return GSA_traj

    def init_GSA_vis_traj(self, batch_size):
        GSA_vis_traj = []
        for i in range(batch_size):
            GSA_vis_traj.append(OrderedDict([
                ('graph_teacher', []),
                ('batch_graphs', []),
                ('current_traj_path', []),
                ('traj_mark', []),
                ('ended', []),
                ('reward', []),
                ('pooling_att', []),
                ('candidate_nodes', []),
                ('updated_visit_graphs', []),
            ]))

        return GSA_vis_traj

    def update_traj(self, traj, new_traj_segment):
        # Save trajectory output and rewards
        for i in range(len(traj)):
            if new_traj_segment[i] != 'none':
                new_path_segment    = [ele for ele in new_traj_segment[i]['path']]
                traj[i]['path']    += new_path_segment
                self.traj_length[i] = self.traj_length[i] + len(new_path_segment)
                traj[i]['length']  += len(new_traj_segment[i])
                traj[i]['feature'] += [ele for ele in new_traj_segment[i]['feature']]
                prev_adj = traj[i]['adj_loc_list'][-1]
                prev_adj = [ele['nextViewpointId'] for ele in prev_adj]
                prev_act_emb = traj[i]['action_embedding'][-1]
                for j in range(len(new_path_segment)):
                    cur_vp   = new_path_segment[j][0]
                    try:
                        action_u = prev_act_emb[prev_adj.index(cur_vp)]
                    except:
                        import pdb; pdb.set_trace()
                    traj[i]['action_u'].append(action_u)
                    prev_adj = new_traj_segment[i]['adj_loc_list'][j]
                    prev_adj = [ele['nextViewpointId'] for ele in prev_adj]
                    prev_act_emb = new_traj_segment[i]['action_embedding'][j]
                traj[i]['adj_loc_list'] += [ele for ele in new_traj_segment[i]['adj_loc_list']]
                traj[i]['action_embedding'] += [ele for ele in new_traj_segment[i]['action_embedding']]

        return traj

    def update_GSA_traj(self, GSA_traj, new_element, last_recorded):
        for k in new_element.keys():
            if new_element[k] is None:
                continue
            for i in range(len(new_element[k])):
                if last_recorded[i] == False:
                    GSA_traj[i][k].append(new_element[k][i])

        return GSA_traj

    def _compute_reward(self, scan, path, gt_path, reward_type, is_last, middle_reward):
        if reward_type == 'sr+sdtw':
            sr  = self.opts.reward_scale * self.env.get_sr(scan, path, gt_path)
            dtw = self.env.get_dtw(scan, path, gt_path, metric='sdtw')
            reward = sr + dtw
        elif reward_type == 'sr+ndtw':
            sr  = self.opts.reward_scale * self.env.get_sr(scan, path, gt_path)
            dtw = self.env.get_dtw(scan, path, gt_path, metric='ndtw')
            reward = sr + dtw
        elif reward_type == 'sr+cls':
            sr  = self.opts.reward_scale * self.env.get_sr(scan, path, gt_path)
            cls = self.env.get_cls(scan, path, gt_path)
            reward = sr + cls
        else:
            raise ValueError('Wrong reward type.')
        if middle_reward == False and (not is_last):
            reward = 0
        return reward

    def update_cur_iteration(self, cur_iteration):
        self.cur_iteration = cur_iteration

    def path_element_from_observation(self, ob):
        return (ob['viewpoint'], ob['heading'], ob['elevation'])

    def realistic_jumping(self, graph, start_step, dest_pos, batch_vp_dict=None):
        #if start_step == self.path_element_from_observation(dest_obs):
        if start_step == dest_pos:
            traj = {'path': [], 'feature': [], 'adj_loc_list': [], 'action_embedding': []}
            return traj
        s = start_step
        t = dest_pos
        path = nx.shortest_path(graph,s,t) #, weight='distance')
        traj_path = [(vp,0,0) for vp in path]
        if batch_vp_dict is not None:
            traj_feat = [batch_vp_dict[ele[0]]['feature'] for ele in traj_path]
            traj_adj  = [batch_vp_dict[ele[0]]['adj_loc_list'] for ele in traj_path]
            traj_act  = [batch_vp_dict[ele[0]]['action_embedding'] for ele in traj_path]
        else:
            traj_feat = []
            traj_adj  = []
            traj_act  = []
        traj = {'path': traj_path, 'feature': traj_feat, 'adj_loc_list': traj_adj, 'action_embedding': traj_act}

        return traj

    def realistic_jumping_action(self, graph, start_step, dest_action, batch_vp_dict=None):
        if start_step[0] == dest_action:
            assert 1 == 0
        s = start_step[0]
        path = nx.shortest_path(graph,s,dest_action) #, weight='distance')
        traj_path = [(vp,0,0) for vp in path[:-1]]
        if batch_vp_dict is not None:
            try:
                traj_feat = [batch_vp_dict[ele[0]]['feature'] for ele in traj_path]
            except:
                import pdb; pdb.set_trace()
            traj_adj  = [batch_vp_dict[ele[0]]['adj_loc_list'] for ele in traj_path]
            traj_act  = [batch_vp_dict[ele[0]]['action_embedding'] for ele in traj_path]
        else:
            traj_feat = []
            traj_adj  = []
            traj_act  = []
            #traj_pre_feat = []
        traj = {'path': traj_path, 'feature': traj_feat, \
                'adj_loc_list': traj_adj, 'action_embedding': traj_act}

        return traj

    def _GSA_env_step(self, world_states, action, ended, scan_id):
        wss, obs, viewpoints  = world_states

        batch_size = len(ended)
        for a_i in range(batch_size):
            action_idx = action[a_i].item()
            if action_idx == 0:
                ended[a_i] = True

        wss = self.env.step(wss, action, obs)
        obs = np.array(self.env.observe(wss)).copy()

        return wss, obs, ended

    def _GSA_env_observe(self, wss, obs, ended):
        pano_img_feat, navigable_feat, \
        viewpoints_indices = super(GraphPanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)

        viewpoints, navigable_index, _ = viewpoints_indices

        navigable_feat, is_valid, _ = self._action_variable(obs)
        navigable_mask = is_valid

        pano_img_feat = pano_img_feat.to(self.device)
        navigable_feat = navigable_feat.to(self.device)
        #target = torch.LongTensor(target_index).to(self.device)

        env_features = (pano_img_feat, navigable_feat, navigable_mask, navigable_index)
        world_states = (wss, obs, viewpoints)

        return env_features, world_states

    def smna_model_step(self, pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                              pre_ctx_attend, navigable_index, ctx_mask):
        u_t_prev = pre_feat
        all_u_t  = navigable_feat
        visual_context = pano_img_feat
        h_0 = h_t
        c_0 = c_t
        ctx = ctx
        ctx_mask = ctx_mask
        batch_size = h_0.shape[0]

        h_1, c_1, \
        attn_text, attn_vision, \
        alpha_text, alpha_action, \
        alpha_vision, action_selector = self.model(u_t_prev, all_u_t, visual_context, h_0, c_0, ctx, ctx_mask, return_action=True)

        return h_1, c_1, alpha_action, action_selector

    def normal_model_step(self, pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                                pre_ctx_attend, navigable_index, ctx_mask):
        h_t, c_t, _, img_attn, ctx_attn, logit, value, navigable_mask, _, action_selector = self.model(
            pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
            None, navigable_index, ctx_mask, return_action_selector)
        return h_1, c_1, logit, action_selector

    def _GSA_model_step(self, ctx_features, env_features, model_states, pre_feat, use_smna_arch):
        # unpack ctx_features
        question, ctx, ctx_mask = ctx_features
        # unpack env states
        pano_img_feat, navigable_feat, navigable_mask, navigable_index = env_features
        # unpack model states
        h_t, c_t = model_states

        # forward pass the network
        if use_smna_arch:
            h_t, c_t, logit, action_selector = self.smna_model_step(pano_img_feat, navigable_feat, pre_feat, 
                                                                    question, h_t, c_t, ctx, None, navigable_index, ctx_mask)
        else:
            h_t, c_t, logit, action_selector = self.normal_model_step(pano_img_feat, navigable_feat, pre_feat, 
                                                                      question, h_t, c_t, ctx, None, navigable_index, ctx_mask)

        logit.data.masked_fill_((navigable_mask == 0).data.bool(), -float('inf'))

        next_model_states  = (h_t, c_t)
        next_action_states = (logit, navigable_mask)

        return next_model_states, next_action_states, action_selector

    def _zip_reverse(self, in_list):
        out_list = [[] for _ in range(len(in_list[0]))]
        for i in range(len(in_list)):
            item = in_list[i]
            for j in range(len(item)):
                out_list[j].append(item[j])
        return out_list

    def _adj_loc_check(self, action_logit, adj_loc_list):
        adj_loc_list_confidence = []
        for i in range(len(adj_loc_list)):
            adj_loc_list_confidence += [action_logit[i][:len(adj_loc_list[i])]]
            assert action_logit[i][:len(adj_loc_list[i])].sum() > -np.inf
        return adj_loc_list_confidence

    def _get_wss_action(self, traj, selected_actions, batch_vp_dict, ended):
        world_states = []
        action       = []
        last_logit   = []
        pre_feat     = torch.zeros(len(traj), self.opts.img_feat_input_dim).to(self.device)
        for i in range(len(traj)):
            path = traj[i]['path']
            last_vp = path[-1][0]
            wss = batch_vp_dict[i][last_vp]['wss']
            if ended[i]:
                action     += [0]
                last_logit += [torch.zeros(1).to(self.device)]
            else:
                adj_loc_list = [ele['nextViewpointId'] for ele in  batch_vp_dict[i][last_vp]['adj_loc_list']]
                action_idx   = adj_loc_list.index(selected_actions[i].split('_')[-1])
                pre_feat[i]  = torch.from_numpy(batch_vp_dict[i][last_vp]['action_embedding'][action_idx]).to(self.device)
                action       += [action_idx]
                last_logit   += [batch_vp_dict[i][last_vp]['adj_loc_list_logit'][action_idx].view(1)]
            world_states += [wss]
        world_states  = self._zip_reverse(world_states)
        for ele1, ele2 in zip(world_states[0], world_states[1]):
            assert ele1.viewpointId == ele2['viewpoint']
        action     = torch.Tensor(action).to(self.device).long()
        last_logit = torch.cat(last_logit, dim=0).unsqueeze(1)
        return world_states, action, pre_feat, last_logit

    def _simulate_jump(self, world_states, obs, static_ctx_features, model_states, jump_path, ended):
        # initialize language information
        batch_size = len(obs)
        question, ctx, ctx_mask = static_ctx_features
        h_t, c_t      = model_states
        segment_ended = ended.copy()

        # step to next node
        action = []
        pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)
        max_jump_len = 0
        for n in range(batch_size):
            if jump_path[n] == 'none' or len(jump_path[n]['path']) == 0:
                action += [0]
                segment_ended[n] = True
                continue
            path = jump_path[n]['path']
            if max_jump_len < len(path):
                max_jump_len = len(path)
            sts_vp = path[0][0]
            adj_loc_list = obs[n]['adj_loc_list']
            adj_loc_list = [ele['nextViewpointId'] for ele in adj_loc_list]
            action_idx   = adj_loc_list.index(sts_vp)
            action      += [action_idx]
            pre_feat[n]  = torch.from_numpy(obs[n]['action_embedding'][action_idx]).to(self.device)
        action = torch.Tensor(action).to(self.device).long()
        world_states = self.env.step(world_states, action, obs)
        obs = np.array(self.env.observe(world_states))

        h_t_final = h_t.clone()
        c_t_final = c_t.clone()
        for step in range(max_jump_len):
            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(GraphPanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            navigable_feat, is_valid, _ = self._action_variable(obs)
            navigable_mask = is_valid

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)

            # forward pass the network
            h_t, c_t, logit, _  = self.smna_model_step(pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                                                       None, None, ctx_mask)

            ended_mask = torch.Tensor(segment_ended.astype(int)).to(self.device).float().unsqueeze(1)
            h_t_final = ended_mask * h_t_final + (1 - ended_mask) * h_t
            c_t_final = ended_mask * c_t_final + (1 - ended_mask) * c_t

            # select action based on prediction
            action = []
            for n in range(batch_size):
                if jump_path[n] != 'none' and step < len(jump_path[n]['path']):
                    action_vp    = jump_path[n]['path'][step][0]
                    adj_loc_list = obs[n]['adj_loc_list']
                    adj_loc_list = [ele['nextViewpointId'] for ele in adj_loc_list]
                    action += [adj_loc_list.index(action_vp)]
                else:
                    action += [0]
                    if jump_path[n] == 'none':
                        assert segment_ended[n] == True
                    else:
                        segment_ended[n] = True
            action = torch.Tensor(action).to(self.device).long()

            # make a viewpoint change in the env
            world_states = self.env.step(world_states, action, obs)
            obs = np.array(self.env.observe(world_states))

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

        return world_states, obs, (h_t_final, c_t_final)

    def rollout_GSA_sac(self, reward_type='sr+ndtw', is_train=True, is_RL=False, return_l2_reg_sum=False, return_vis=False):
        if self.opts.use_smna_arch:
            wss = self.env.reset(sort=True)  # load a mini-batch
            obs = np.array(self.env.observe(wss)).copy()
            batch_size = len(obs)
            
            seq, seq_mask, seq_lengths = self._FAST_proc_batch(obs)
            ctx_mask = seq_mask
            
            ctx, h_t, c_t = self.encoder(seq, seq_lengths)
            question = h_t.clone()
        else:
            assert 1 == 0

        pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

        # init GSA trajectories
        GSA_traj = self.init_GSA_traj(batch_size)
        GSA_vis_traj = self.init_GSA_vis_traj(batch_size)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        # Initialize mental map
        visit_graphs = [nx.DiGraph() for _ in range(batch_size)]
        for ob, g in zip(obs, visit_graphs): g.add_node(ob['viewpoint'])

        # Initialize Queue
        static_ctx_features = (question.clone(), ctx.clone(), ctx_mask.clone())
        env_features, world_states  = self._GSA_env_observe(wss, obs, ended)
        model_states, action_states, action_selector = self._GSA_model_step(ctx_features=static_ctx_features, \
                                                                                           env_features=env_features, \
                                                                                           model_states=(h_t, c_t), \
                                                                                           pre_feat=pre_feat, \
                                                                                           use_smna_arch=self.opts.use_smna_arch)

        # Initialize dictionary for saving observations
        #adj_loc_list_logit = action_states[0] #-> action logits
        #adj_loc_list_logit = self._adj_loc_check(adj_loc_list_logit, [obs[i]['adj_loc_list'] for i in range(len(obs))])
        batch_vp_dict = [{obs[i]['viewpoint']: {'feature': obs[i]['feature'], \
                                                'start_pos': 1, \
                                                'adj_loc_list': obs[i]['adj_loc_list'], \
                                                #'adj_loc_list_logit': adj_loc_list_logit[i], \
                                                'adj_loc_list_logit': None, \
                                                'adj_loc_list_active': [True for _ in range(len(obs[i]['adj_loc_list']))], \
                                                'h': model_states[0][i], \
                                                'last_logit': torch.zeros(1).to(self.device), \
                                                'wss': (world_states[0][i], world_states[1][i], world_states[2][i]), \
                                                'visited_step': 0, \
                                                'action_embedding': obs[i]['action_embedding']}} for i in range(batch_size)]

        # Rollout with Graph Search
        loss          = 0
        actor_entropy = 0
        for step in range(self.opts.max_episode_len):
            # Remember to not repeat the previous traversed actions/partial trajs
            num_mp_steps = self.opts.max_mp_steps

            jump_function, model_function = None, None
            if self.opts.use_moving_window:
                cur_step = step
            else:
                cur_step = None
            batch_policies, batch_actions, batch_logits, actor_ent, \
                selected_actions, candidate_nodes, updated_visit_graphs, \
                   l2_reg_sum, adj_loc_list_logit, v_matrices = self.actor_network(obs, \
                                                                                   jump_function, \
                                                                                   model_function, \
                                                                                   model_states, \
                                                                                   action_states, \
                                                                                   batch_vp_dict, \
                                                                                   [ele.copy() for ele in visit_graphs], \
                                                                                   num_mp_steps, \
                                                                                   action_selector, \
                                                                                   question, \
                                                                                   self.opts.GSA_top_K, \
                                                                                   self.opts.separate_query, \
                                                                                   ended=ended, \
                                                                                   graph_teacher_option=self.opts.graph_teacher_option, \
                                                                                   is_train=is_train, \
                                                                                   ctx=ctx, \
                                                                                   ctx_mask=ctx_mask, \
                                                                                   cur_step=cur_step, \
                                                                                   window_size=self.opts.moving_window_size
                                                                )

            adj_loc_list_logit = self._adj_loc_check(adj_loc_list_logit, [obs[i]['adj_loc_list'] for i in range(len(obs))])
            for i in range(len(batch_vp_dict)):
                if batch_vp_dict[i][obs[i]['viewpoint']]['adj_loc_list_logit'] is None:
                    batch_vp_dict[i][obs[i]['viewpoint']]['adj_loc_list_logit'] = adj_loc_list_logit[i]

            batch_graph_teacher = self._get_graph_teacher(obs, \
                                                          candidate_nodes, \
                                                          option=self.opts.graph_teacher_option, \
                                                          logits=batch_logits, \
                                                          updated_visit_graphs=updated_visit_graphs, \
                                                          separate_query=self.opts.separate_query, \
                                                          ended=ended, \
                                                          traj=traj, \
                                                          batch_vp_dict=batch_vp_dict, \
                                                          default_teacher=[ele['teacher_action'] for ele in obs], \
                                                          allow_missing=is_train==False)

            actor_entropy += actor_ent
            current_traj_path = [ele['path'].copy() for ele in traj]
            GSA_traj = self.update_GSA_traj(GSA_traj, {'policy': batch_policies, \
                                                       'action': batch_actions,  \
                                                       'logit':  batch_logits, \
                                                       'graph_teacher':  batch_graph_teacher.copy(), \
                                                       'current_traj_path': current_traj_path, \
                                                       'pooling_att': v_matrices[0]}, last_recorded)

            batch_graphs = [self.env.graphs[obs[i]['scan']] for i in range(len(obs))]
            GSA_vis_traj = self.update_GSA_traj(GSA_vis_traj, {'batch_graphs': batch_graphs, \
                                                               'graph_teacher':  batch_graph_teacher.copy(), \
                                                               'current_traj_path': current_traj_path, \
                                                               'pooling_att': v_matrices[0].detach().cpu().numpy(), \
                                                               'candidate_nodes': candidate_nodes, \
                                                               'updated_visit_graphs': updated_visit_graphs}, last_recorded)

            if is_train and (not is_RL) and self.actor_network.feedback != 'sample' and self.actor_network.feedback != 'argmax':
                deviation = int(''.join([n for n in self.actor_network.feedback if n.isdigit()]))
                for i,ob in enumerate(obs):
                    if ended[i]:
                        continue
                    if ob['deviation'] >= deviation:
                        batch_actions[i]    = batch_graph_teacher[i]
                        selected_actions[i] = candidate_nodes[i][batch_graph_teacher[i]][6:]

            # Add the teleport traj segments to trajs
            new_traj_segment = []
            for i, _ in enumerate(obs):
                if not ended[i]:
                    last_vp = traj[i]['path'][-1]
                    selected_action = 'query_' + selected_actions[i]
                    jump_segment = self.realistic_jumping_action(updated_visit_graphs[i], last_vp, selected_action, batch_vp_dict[i])
                    jump_segment = {k:jump_segment[k][1:] for k in jump_segment.keys()}
                    if self.opts.GSA_top_K == 0:
                        if not self.opts.graph_teacher_option == 'follow_gt':
                            assert len(jump_segment['path']) == 0, len(jump_segment['path'])
                    new_traj_segment.append(jump_segment)
                else:
                    new_traj_segment.append('none')

            traj = self.update_traj(traj, new_traj_segment)

            # remove the already selected path candidates from active list
            for i in range(len(batch_vp_dict)):
                if not ended[i]:
                    if self.opts.separate_query:
                        selected_action = 'query_' + selected_actions[i].split("_")[1]
                        neighbors       = [selected_actions[i].split("_")[0]]
                    else:
                        selected_action = 'query_' + selected_actions[i]
                        #neighbors       = updated_visit_graphs[i].predecessors(selected_action)
                        neighbors       = [traj[i]['path'][-1][0]]
                    for neighbor in neighbors:
                        adj_loc_list = [ele['nextViewpointId'] for ele in batch_vp_dict[i][neighbor]['adj_loc_list']]
                        action_idx   = adj_loc_list.index(selected_action[6:])
                        batch_vp_dict[i][neighbor]['adj_loc_list_active'][action_idx] = False

            # run model through jump path
            _, _,  model_states   = self._simulate_jump(world_states[0], \
                                                        world_states[1],\
                                                        static_ctx_features, \
                                                        model_states, \
                                                        new_traj_segment, \
                                                        ended)

            world_states_jumped, action, pre_feat, last_logit = self._get_wss_action(traj, selected_actions, batch_vp_dict, ended)
            wss, obs, ended = self._GSA_env_step(world_states_jumped, action, ended, scan_id)

            new_traj_step = []
            obs_jumped = world_states_jumped[1]
            for i in range(batch_size):
                if not ended[i]:
                    single_step = {}
                    single_step['path']    = [self.path_element_from_observation(obs[i])]
                    for ss_key in ['feature', 'adj_loc_list', 'action_embedding']:
                        single_step[ss_key] = [obs[i][ss_key]]
                    new_traj_step.append(single_step)
                    if not visit_graphs[i].has_edge(obs_jumped[i]['viewpoint'], obs[i]['viewpoint']):
                        edge_dis = self.env.distances[obs[i]['scan']][obs_jumped[i]['viewpoint']][obs[i]['viewpoint']]
                        visit_graphs[i].add_edge(obs_jumped[i]['viewpoint'], obs[i]['viewpoint'], weight=1, distance=edge_dis)
                        visit_graphs[i].add_edge(obs[i]['viewpoint'], obs_jumped[i]['viewpoint'], weight=2, distance=edge_dis)
                else:
                    new_traj_step.append('none')

            traj = self.update_traj(traj, new_traj_step)
            GSA_traj = self.update_GSA_traj(GSA_traj, {'traj_mark': [len(traj_ele['path']) for traj_ele in traj], \
                                                       'ended': ended}, last_recorded)

            # Expand on new nodes
            env_features, world_states  = self._GSA_env_observe(wss, obs, ended)
            model_states, action_states, action_selector = self._GSA_model_step(ctx_features=static_ctx_features, \
                                                                                               env_features=env_features, \
                                                                                               model_states=model_states, \
                                                                                               pre_feat=pre_feat, \
                                                                                               use_smna_arch=self.opts.use_smna_arch)

            # Update batch_vp_dict
            #adj_loc_list_logit = action_states[0]
            #adj_loc_list_logit = self._adj_loc_check(adj_loc_list_logit, [obs[i]['adj_loc_list'] for i in range(len(obs))])
            for i in range(batch_size):
                if not last_recorded[i]:
                    if ended[i] == True and last_recorded[i] == False:
                        last_recorded[i] = True
                        assert obs[i]['viewpoint'] in batch_vp_dict[i].keys()
                    if not (obs[i]['viewpoint'] in batch_vp_dict[i].keys()):
                        batch_vp_dict[i][obs[i]['viewpoint']] = {'feature': obs[i]['feature'], \
                                                                 'start_pos': 0, \
                                                                 'adj_loc_list': obs[i]['adj_loc_list'], \
                                                                 #'adj_loc_list_logit': adj_loc_list_logit[i], \
                                                                 'adj_loc_list_logit': None, \
                                                                 'adj_loc_list_active': [True] * len(obs[i]['adj_loc_list']), \
                                                                 'h': model_states[0][i], \
                                                                 'last_logit': last_logit[i], \
                                                                 'wss': (world_states[0][i], world_states[1][i], world_states[2][i]), \
                                                                 'visited_step': step + 1, \
                                                                 'action_embedding': obs[i]['action_embedding']}

        self.dist_from_goal = self.get_distance(traj)

        results = [None, traj, GSA_traj, actor_entropy, batch_vp_dict, \
                      (seq.cpu().numpy(), ctx_mask.cpu().numpy(), seq_lengths), (h_t, ctx, ctx_mask)]

        if return_l2_reg_sum:
            results += [l2_reg_sum]

        if return_vis:
            results += [GSA_vis_traj]

        return results

    def get_supervised_loss(self, GSA_traj):
        loss = 0
        for i, ele in enumerate(GSA_traj):
            logit = ele['logit']
            graph_teacher = ele['graph_teacher']
            for l, gt in zip(logit, graph_teacher):
                assert gt < len(l)
                loss += F.cross_entropy(l.unsqueeze(0), torch.Tensor([gt]).to(self.device).long())
        loss /= len(GSA_traj)
        return loss

    def compare_path(self, teacher, path):
        count = 0
        min_len = min(len(teacher), len(path))
        for i in range(min_len):
            if path[i] == teacher[i]:
                count += 1
            else:
                break
        return count

    def compare_path_last(self, teacher, node):
        if node not in teacher:
            return -1
        else:
            return teacher.index(node)

    def _get_graph_teacher(self, obs, candidate_nodes, option, logits, updated_visit_graphs=None, separate_query=False, ended=None, traj=None, batch_vp_dict=None, default_teacher=[], allow_missing=False):
        batch_graph_teacher = []
        if option == 'follow_gt' or option == 'follow_gt_tf':
            for i, ob in enumerate(obs):
                if ended[i]:
                    batch_graph_teacher += [0]
                    continue
                teacher = ob['teacher'] + [ob['teacher'][-1]]
                max_ovlp = -1
                gt_k     = None
                for k, node in enumerate(candidate_nodes[i]):
                    if 'query_' not in node:
                        continue
                    if separate_query:
                        node = node.split('_')[-1]
                    else:
                        node = node[6:]
                    path = [ele[0] for ele in traj[i]['path'][:-1]] + \
                           self.env.paths[ob['scan']][ob['viewpoint']][node]
                    if self.opts.dataset_name == 'R4R':
                        # dtw
                        ovlp = self.env.get_dtw(ob['scan'], path, teacher, metric='ndtw')
                    else:
                        # last node selection
                        ovlp = self.compare_path_last(teacher, node)
                    # direct path compare
                    # ovlp = self.compare_path(teacher, path)
                    if max_ovlp < ovlp:
                        max_ovlp = ovlp
                        gt_k = k
                if not allow_missing:
                    assert gt_k >= 0
                else:
                    if gt_k is None:
                        gt_k = 0
                assert gt_k < len(candidate_nodes[i])
                assert gt_k < len(logits[i])
                batch_graph_teacher += [gt_k]
        elif option == 'best_candidate':
            for i in range(len(obs)):
                if ended[i]:
                    batch_graph_teacher += [0]
                    continue
                goal_viewpoint = obs[i]['teacher'][-1]
                min_dis  = 10000
                gt_k     = None
                for k, node in enumerate(candidate_nodes[i]):
                    if 'query_' not in node: 
                        continue
                    if separate_query:
                        src, node = node.split('_')[1:]
                    else:
                        node = node[6:]
                    teacher = self.env.paths[obs[i]['scan']][node][goal_viewpoint]
                    dis     = len(teacher)
                    if separate_query:
                        if node == src:
                            if node != goal_viewpoint:
                                dis = 10000
                    if min_dis > dis:
                        min_dis = dis
                        gt_k = k
                assert gt_k >= 0
                assert gt_k < len(candidate_nodes[i])
                assert gt_k < len(logits[i])
                batch_graph_teacher += [gt_k]
        elif option == 'best_candidate_joint':
            for i in range(len(obs)):
                if ended[i]:
                    batch_graph_teacher += [0]
                    continue
                goal_viewpoint = obs[i]['teacher'][-1]
                min_dis  = 10000
                gt_k     = None
                for k, node in enumerate(candidate_nodes[i]):
                    if 'query_' not in node:
                        continue
                    jump_path = self.realistic_jumping_action(updated_visit_graphs[i], [obs[i]['viewpoint']], node)
                    if separate_query:
                        src, node = node.split('_')[1:]
                    else:
                        node = node[6:]
                    teacher = self.env.paths[obs[i]['scan']][node][goal_viewpoint]
                    dis     = len(jump_path['path']) - 1 + len(teacher)
                    if separate_query:
                        if src == node:
                            if node != goal_viewpoint:
                                dis = 10000
                    if min_dis > dis:
                        min_dis = dis
                        gt_k = k
                try:
                    assert gt_k >= 0
                except:
                    import pdb; pdb.set_trace()
                #if obs[i]['adj_loc_list'][default_teacher[i]]['nextViewpointId'] != candidate_nodes[i][gt_k].split('_')[-1]:
                #    import pdb; pdb.set_trace()
                assert gt_k < len(candidate_nodes[i])
                assert gt_k < len(logits[i])
                batch_graph_teacher += [gt_k]
        elif option == 'default_teacher':
            for i in range(len(obs)):
                if ended[i]:
                    batch_graph_teacher += [0]
                    continue
                goal_viewpoint = obs[i]['teacher'][-1]
                min_dis  = 10000
                gt_k     = None
                for k, node in enumerate(candidate_nodes[i]):
                    if 'query_' not in node:
                        continue
                    if separate_query:
                        src, node = node.split('_')[1:]
                    else:
                        node = node[6:]
                    if obs[i]['adj_loc_list'][default_teacher[i]]['nextViewpointId'] == node:
                        gt_k = k
                try:
                    assert gt_k >= 0
                except:
                    import pdb; pdb.set_trace()
                if obs[i]['adj_loc_list'][default_teacher[i]]['nextViewpointId'] != candidate_nodes[i][gt_k].split('_')[-1]:
                    import pdb; pdb.set_trace()
                assert gt_k < len(candidate_nodes[i])
                assert gt_k < len(logits[i])
                batch_graph_teacher += [gt_k]
        else:
            raise ValueError('Not implemented.')
        return batch_graph_teacher

    def get_distance(self, traj):
        distances = []
        for ele in traj:
            dist = self.env.distances[ele['scan']][ele['path'][-1][0]][ele['teacher'][-1]]
            distances += [dist]
        return distances

    def FAST_batch_instructions_from_encoded(self, encoded_instructions, max_length, reverse=False, sort=False, tok=None, addEos=True):
        # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
        # make sure pad does not start any sentence
        num_instructions = len(encoded_instructions)
        base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']
        vocab_pad_idx = base_vocab.index('<PAD>')
        vocab_eos_idx = base_vocab.index('<EOS>')
        seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
        seq_lengths = []
        inst_mask = []
       
        for i, inst in enumerate(encoded_instructions):
            if len(inst) > 0:
                assert inst[-1] != vocab_eos_idx
            if reverse:
                inst = inst[::-1]
            if addEos:
                inst = np.concatenate((inst, [vocab_eos_idx]))
            inst = inst[:max_length]
            if tok:
                inst_mask.append(tok.filter_verb(inst,sel_verb=False)[1])
            seq_tensor[i,:len(inst)] = inst
            seq_lengths.append(len(inst))
        
        seq_tensor = torch.from_numpy(seq_tensor)
        if sort:
            seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)
            seq_lengths = list(seq_lengths)
            seq_tensor = seq_tensor[perm_idx]
        else:
            perm_idx = np.arange((num_instructions))
        
        mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]
        
        if tok:
            for i,idx in enumerate(perm_idx):
                mask[i][inst_mask[idx]] = 1
        
        ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
                 try_cuda(mask.byte()), \
                 seq_lengths
        if sort:
            ret_tp = ret_tp + (list(perm_idx),)

        return ret_tp

    def _FAST_proc_batch(self, obs, beamed=False):
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        # tok = self.env.tokenizer if self.attn_only_verb else None
        tok = None
        max_instruction_length = 80
        reverse_instruction = True
        return self.FAST_batch_instructions_from_encoded(encoded_instructions, max_instruction_length, reverse=reverse_instruction, tok=tok)

    def _feature_variables(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        feature_lists = list(zip(*[ob['feature'] for ob in (flatten(obs) if beamed else obs)]))
        batched_feature = torch.zeros(len(feature_lists[0]), len(feature_lists), len(feature_lists[0][0]))
        for i, feature_list in enumerate(feature_lists):
            for j, feature in enumerate(feature_lists[i]):
                batched_feature[j][i] = torch.from_numpy(feature)
        return batched_feature

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros(
            (len(obs), max_num_a, action_embedding_dim),
            dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            for n_a, adj_dict in enumerate(adj_loc_list):
                action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (
            Variable(torch.from_numpy(action_embeddings), requires_grad=False).cuda(),
            Variable(torch.from_numpy(is_valid), requires_grad=False).cuda(),
            is_valid)
