import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time
import sys

import torch.nn.functional as F
import torch.distributions as D

from models.modules import VisualSoftDotAttention
from utils import Entropy
from graphlib.EdgeGraph.GraphLayer  import GraphLayer
from smna_models.model import WhSoftDotAttention
from smna_models.model import WhSoftDotAttentionMatrix

import random

class GraphPolicyNetwork(nn.Module):
    """ A policy network used for Graph Search. """

    def __init__(self, policy_mapping, policy_indim=-1, feature_dim=512, emb_dim=128, \
                       graph_dim=48, node_history=True, edge_history=False, use_global=True, \
                       use_attention=True, message_normalize=False, no_h_feature=True, feature_pool='max', \
                       action_query_dim=-1, agent_rnn_size=512, visual_mlp=None, use_all_query=False, \
                       use_cur_pos_h=False, graph_pooling_config=[False, 6, 32, 64, 3, 'softmax', 3], \
                       graph_attention_type='sigmoid', use_ctx_attend=False, \
                       gnn_dropout=0):
        super(GraphPolicyNetwork, self).__init__()
        self.feedback = 'sample'
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_pool  = feature_pool
        self.feature_dim   = feature_dim
        self.use_all_query = use_all_query
        self.use_cur_pos_h = use_cur_pos_h

        self.set_up_graph_pooling_config(graph_pooling_config)

        self.hidden_dim    = 1024

        self.query_map     = nn.Linear(1, 32)
        self.start_pos_map = nn.Linear(1, 32)

        num_feat = 2
        self.init_node      = nn.Linear(self.hidden_dim + num_feat * 32 + agent_rnn_size + agent_rnn_size, graph_dim)
        self.init_query     = nn.Linear(self.hidden_dim, graph_dim)
        self.init_edge      = nn.Linear(self.hidden_dim, graph_dim)
        if policy_mapping == 'feature_only':
            self.init_node      = nn.Linear(graph_dim, graph_dim)
        self.output_net     = nn.Linear(graph_dim, 1)
        if visual_mlp is None:
            self.visual_mlp = nn.Sequential(
                              nn.Linear(feature_dim, self.hidden_dim),
                              nn.Dropout(p=0.5),
                              nn.ReLU()
                          )
            self.extra_visual_linear = None
        else:
            self.visual_mlp = visual_mlp
        #self.extra_visual_linear = nn.Linear(self.hidden_dim, graph_dim)
        self.extra_visual_linear    = None
        self.action_attention_layer = WhSoftDotAttention(action_query_dim, graph_dim, no_mapping=False)
        num_functions      = 3
        self.num_functions = num_functions

        if self.use_graph_pooling:
            pooling_output_dim     = self.pooling_num_node
            self.pooling_mapping   = []
            self.pooling_mapping_2 = []
            self.planner_reverse_mapping = []
            self.planner_pooling   = []
            for i in range(self.pooling_channels):
                self.pooling_mapping   += [nn.Linear(graph_dim, self.pooling_graph_dim)]
                self.pooling_mapping_2 += [nn.Linear(self.pooling_graph_dim, pooling_output_dim)]
                self.planner_reverse_mapping += [nn.Linear(self.planner_graph_dim, graph_dim)]
                self.planner_pooling += [GraphLayer(nodedims=[self.pooling_graph_dim, self.pooling_graph_dim], \
                                                    edgedims=[self.pooling_graph_dim, self.pooling_graph_dim], \
                                                    globaldims=[self.pooling_graph_dim, self.pooling_graph_dim], \
                                                    node_history=True, \
                                                    edge_history=False, \
                                                    use_global=True, \
                                                    num_functions=num_functions, \
                                                    use_attention=use_attention, \
                                                    attention_type=graph_attention_type, \
                                                    message_normalize=message_normalize, \
                                                    dropout=0)]
            self.pooling_mapping   = nn.ModuleList(self.pooling_mapping)
            self.pooling_mapping_2 = nn.ModuleList(self.pooling_mapping_2)
            self.planner_reverse_mapping = nn.ModuleList(self.planner_reverse_mapping)
            self.planner_pooling         = nn.ModuleList(self.planner_pooling)

            self.planner_mapping = []
            self.graph_net       = []
            for i in range(self.pooling_channels):
                self.planner_mapping += [nn.Linear(graph_dim, self.planner_graph_dim)]
                self.graph_net += [GraphLayer(nodedims=[self.planner_graph_dim, self.planner_graph_dim], \
                                              edgedims=[self.planner_graph_dim, self.planner_graph_dim], \
                                              globaldims=[self.planner_graph_dim, self.planner_graph_dim], \
                                              node_history=True, \
                                              edge_history=False, \
                                              use_global=True, \
                                              num_functions=num_functions, \
                                              use_attention=use_attention, \
                                              message_normalize=message_normalize, \
                                              dropout=gnn_dropout)]
            self.graph_net       = nn.ModuleList(self.graph_net)
            self.planner_mapping = nn.ModuleList(self.planner_mapping)

            # attend to ctx
            if use_ctx_attend:
                self.ctx_attend     = []
                self.ctx_attend_map = []
                for i in range(self.pooling_channels):
                    self.ctx_attend += [WhSoftDotAttentionMatrix(h_dim=self.planner_graph_dim, \
                                                                 v_dim=agent_rnn_size, \
                                                                 no_mapping=False, \
                                                                 reverse_mapping=True)]
                    self.ctx_attend_map += [nn.Linear(self.planner_graph_dim * 2, self.planner_graph_dim)]
                self.ctx_attend     = nn.ModuleList(self.ctx_attend)
                self.ctx_attend_map = nn.ModuleList(self.ctx_attend_map)
            else:
                self.ctx_attend = None
        else:
            self.graph_net     = GraphLayer(nodedims=[graph_dim, graph_dim], \
                                            edgedims=[graph_dim, graph_dim], \
                                            globaldims=[graph_dim, graph_dim], \
                                            node_history=True, \
                                            edge_history=False, \
                                            use_global=True, \
                                            num_functions=num_functions, \
                                            use_attention=use_attention, \
                                            message_normalize=message_normalize, \
                                            dropout=gnn_dropout)

        self.entropy = Entropy(size_average=True)
        self.graph_dim = graph_dim
        self.policy_mapping = policy_mapping

    def set_up_graph_pooling_config(self, graph_pooling_config):
        self.use_graph_pooling = graph_pooling_config[0]
        self.pooling_num_node  = graph_pooling_config[1]
        self.pooling_graph_dim = graph_pooling_config[2]
        self.planner_graph_dim = graph_pooling_config[3]
        self.pooling_mp_steps  = graph_pooling_config[4]
        self.normalize_pooling = graph_pooling_config[5]
        self.pooling_channels  = graph_pooling_config[6]

    def _get_edge(self, node_pairs, batch_vp_dict):
        if 'query' in node_pairs[0] or 'query' in node_pairs[1]:
            query_node    = node_pairs[0] if 'query' in node_pairs[0] else node_pairs[1]
            position_node = node_pairs[1] if 'query' in node_pairs[0] else node_pairs[0]
            query_node    = query_node[6:]
        else:
            position_node, query_node = node_pairs
        node_adj = batch_vp_dict[position_node]['adj_loc_list']
        node_adj = [ele['nextViewpointId'] for ele in node_adj]
        action_embedding = batch_vp_dict[position_node]['action_embedding'][node_adj.index(query_node)]
        #action_embedding = torch.from_numpy(action_embedding).to(self.device).unsqueeze(0)
        action_embedding = action_embedding[np.newaxis,...]
        return action_embedding

    def _get_adj_loc_idx(self, vp_dict, node):
        adj_loc_list  = [ele['nextViewpointId'] for ele in vp_dict['adj_loc_list']]
        query_tgt_idx = adj_loc_list.index(node)
        #query_tgt_emb = torch.from_numpy(vp_dict['action_embedding'][query_tgt_idx]).to(self.device).unsqueeze(0)
        query_tgt_emb = vp_dict['action_embedding'][query_tgt_idx][np.newaxis,...]
        return query_tgt_idx, query_tgt_emb

    def _prepare_graph_feat(
            self, obs, batch_vp_dict, visit_graphs, top_K=None, ended=None, graph_teacher_option='', is_train=False, cur_step=None, window_size=None):
        batch_size    = len(batch_vp_dict)
        node_feat_all = []
        node_feat_idx = []
        edge_feat_all = []
        edge_feat_idx = []
        E_matrix_all  = []
        query_set_all = []
        graph_node_all = []
        start_pos_all  = []
        cur_pos_all    = []

        max_num_node = -1
        global_node_count = 0
        avg_missing_rate = 0
        for i in range(batch_size):
            # update visit graph
            visited_graph_node     = list(visit_graphs[i].nodes.data())
            visited_graph_node     = [ele[0] for ele in visited_graph_node]
            missing_count = 0
            for k in range(len(visited_graph_node)):
                node_vp = visited_graph_node[k]
                if cur_step is not None:
                    if cur_step - batch_vp_dict[i][node_vp]['visited_step'] >= window_size \
                       and node_vp != obs[i]['viewpoint'] \
                       and (visit_graphs[i].has_edge(node_vp, obs[i]['viewpoint']) \
                            or visit_graphs[i].has_edge(obs[i]['viewpoint'], node_vp)):
                        continue
                adj_loc_list        = [ele['nextViewpointId'] for ele in batch_vp_dict[i][node_vp]['adj_loc_list']]
                adj_loc_list_logit  = batch_vp_dict[i][node_vp]['adj_loc_list_logit']
                if adj_loc_list_logit is None:
                    assert node_vp == obs[i]['viewpoint']
                    adj_loc_list_logit = np.ones(len(batch_vp_dict[i][node_vp]['adj_loc_list']))
                else:
                    pass
                #adj_loc_list_logit = np.random.rand(len(batch_vp_dict[i][node_vp]['adj_loc_list']))
                if isinstance(adj_loc_list_logit, torch.Tensor):
                    adj_loc_list_logit = adj_loc_list_logit.detach().cpu().numpy()
                if self.use_all_query:
                    adj_loc_list_active = [True] * len(batch_vp_dict[i][node_vp]['adj_loc_list_active'])
                else:
                    adj_loc_list_active = batch_vp_dict[i][node_vp]['adj_loc_list_active']
                active_indicator    = (np.array(adj_loc_list_active)==True)
                sorted_logit        = np.argsort(adj_loc_list_logit[active_indicator])[::-1]
                thresh_logit        = 10e6
                if len(sorted_logit) > 0:
                    thresh_logit    = adj_loc_list_logit[active_indicator][sorted_logit[max(0, min(top_K, len(sorted_logit)) - 1)]]
                if top_K is 0:
                    thresh_logit    = 10e6
                for k_loc in range(len(adj_loc_list)):
                    action_vp = 'query_' + adj_loc_list[k_loc]
                    if node_vp != obs[i]['viewpoint']:
                        if graph_teacher_option == 'follow_gt' and is_train:
                            if adj_loc_list_active[k_loc] == False:
                                continue
                            if adj_loc_list_logit[k_loc] < thresh_logit:
                                if action_vp[6:] in obs[i]['teacher']:
                                    missing_count += 1
                                else:
                                    continue
                        else:
                            if adj_loc_list_active[k_loc] == False:
                                continue
                            if adj_loc_list_logit[k_loc] < thresh_logit:
                                if action_vp[6:] in obs[i]['teacher']:
                                    missing_count += 1
                                continue
                    visit_graphs[i].add_edge(node_vp, action_vp, weight=1, distance=1)
                    #visit_graphs[i].add_edge(action_vp, node_vp, weight=2, distance=1)

            # get all node
            new_graph_node   = [ele[0] for ele in list(visit_graphs[i].nodes.data())]
            if graph_teacher_option == 'follow_gt' and False:
                print('missing nodes v.s. total nodes: ', missing_count, len(new_graph_node))
                avg_missing_rate += float(missing_count) / float(len(new_graph_node))
            elif False:
                print('missing nodes v.s. total nodes: ', missing_count, len(new_graph_node) + missing_count)
                avg_missing_rate += float(missing_count) / (float(len(new_graph_node) + missing_count))
            new_graph_edge   = list(visit_graphs[i].edges.data())
            num_node         = len(new_graph_node)
            if max_num_node < num_node:
                max_num_node = num_node
            graph_node_all  += [new_graph_node]

            # get graph feat
            query_set          = []
            start_pos          = []
            cur_pos            = []

            for k in range(num_node):
                if 'query_' in new_graph_node[k]:
                    neighbors = visit_graphs[i].predecessors(new_graph_node[k])
                    for neighbor in neighbors:
                        neighbor_idx, neighbor_emb = self._get_adj_loc_idx(batch_vp_dict[i][neighbor], new_graph_node[k][6:])
                        node_feat_all += [neighbor_emb]
                        node_feat_idx += [global_node_count]
                        edge_feat_idx += [-1]
                    query_set += [1]
                    start_pos += [0]
                    cur_pos   += [0]
                else:
                    node_feature = batch_vp_dict[i][new_graph_node[k]]['action_embedding']
                    #node_feature = torch.from_numpy(node_feature).to(self.device)
                    node_feat_all += [node_feature]
                    node_feat_idx += [global_node_count] * node_feature.shape[0]
                    edge_feat_idx += [-1] * node_feature.shape[0]
                    query_set     += [0]
                    start_pos     += [batch_vp_dict[i][new_graph_node[k]]['start_pos']]
                    cur_pos       += [int(new_graph_node[k]==obs[i]['viewpoint'])]
                global_node_count += 1

            assert len(query_set) > 0
            assert len(start_pos) == len(cur_pos)
            query_set_all += [np.array(query_set)]
            start_pos_all   += [np.array(start_pos)]
            cur_pos_all     += [np.array(cur_pos)]

            #adj        = torch.zeros(num_node, num_node).to(self.device)
            adj        = np.zeros([num_node, num_node])
            graph_edge = list(visit_graphs[i].edges.data())
            edge_feat  = [[[] for _ in range(num_node)] for _ in range(num_node)]
            for k, edge in enumerate(graph_edge):
                src = new_graph_node.index(edge[0])
                tgt = new_graph_node.index(edge[1])
                adj[src,tgt] = int(edge[2]['weight'])
                edge_feat[src][tgt] = [self._get_edge([edge[0], edge[1]], batch_vp_dict[i])]
                edge_feat_idx += [i]
                node_feat_idx += [-1]
            for ei in range(num_node):
                for ej in range(num_node):
                    node_feat_all += edge_feat[ei][ej]
            E_matrix_all    += [adj]

        avg_missing_rate = avg_missing_rate / batch_size
        if graph_teacher_option == 'follow_gt' and False:
            print('Averate missing rate: ', avg_missing_rate)

        node_feat_all = np.concatenate(node_feat_all, axis=0)
        node_feat_all = torch.from_numpy(node_feat_all).to(self.device)
        node_feat_all = self.visual_mlp(node_feat_all)
        if self.extra_visual_linear is not None:
            node_feat_all = self.extra_visual_linear(node_feat_all)
        assert len(edge_feat_idx) == len(node_feat_idx)
        assert node_feat_all.shape[0] == len(node_feat_idx)

        # Concat everything into torch tensors
        node_feat = torch.zeros(batch_size, max_num_node, self.hidden_dim).to(self.device)
        node_feat_idx = torch.Tensor(node_feat_idx).to(self.device)
        global_node_count = 0
        for i, graph_node in enumerate(graph_node_all):
            for j, node in enumerate(graph_node):
                if self.feature_pool == 'max':
                    node_feat[i,j] = node_feat_all[node_feat_idx==global_node_count].max(dim=0)[0]
                elif self.feature_pool == 'mean':
                    node_feat[i,j] = node_feat_all[node_feat_idx==global_node_count].mean(dim=0)
                elif self.feature_pool == 'sum':
                    node_feat[i,j] = node_feat_all[node_feat_idx==global_node_count].sum(dim=0)
                global_node_count += 1

        edge_feat = torch.zeros(batch_size, max_num_node, max_num_node, self.hidden_dim).to(self.device)
        edge_feat_idx = torch.Tensor(edge_feat_idx).to(self.device)
        E_matrix_all  = [torch.from_numpy(ele).to(self.device) for ele in E_matrix_all]
        for i, E in enumerate(E_matrix_all):
            num_node = E.shape[0]
            E_full   = torch.zeros(max_num_node, max_num_node).to(self.device)
            E_full[:num_node,:num_node] = E
            edge_feat[i,E_full>0] = node_feat_all[edge_feat_idx==i]

        E_matrix  = torch.zeros(batch_size, max_num_node, max_num_node).to(self.device)
        E_matrix_function = torch.zeros(batch_size, max_num_node, max_num_node, self.num_functions).to(self.device)
        query     = torch.zeros(batch_size, max_num_node).to(self.device)
        start_pos = torch.zeros(batch_size, max_num_node).to(self.device)
        cur_pos   = torch.zeros(batch_size, max_num_node).to(self.device)
        node_mask = torch.zeros(batch_size, max_num_node).to(self.device)
        E_matrix_function[:,:,:,0] = 1

        for i in range(batch_size):
            num_node  = E_matrix_all[i].shape[-1]
            E_matrix[i,:num_node,:num_node] = E_matrix_all[i]
            if self.num_functions == 4:
                E_matrix_function[i,:num_node,:num_node,3] = 1 - torch.eye(num_node).to(self.device).long()
            E_matrix_function[i,:num_node,:num_node,2] = (E_matrix_all[i] == 2).long()
            E_matrix_function[i,:num_node,:num_node,1] = (E_matrix_all[i] == 1).long()
            E_matrix_function[i,:num_node,:num_node,0] = 1 - ((E_matrix_all[i]) > 0).float()
            query[i,:num_node]     = torch.from_numpy(query_set_all[i]).to(self.device)
            start_pos[i,:num_node] = torch.from_numpy(start_pos_all[i]).to(self.device)
            cur_pos[i,:num_node]   = torch.from_numpy(cur_pos_all[i]).to(self.device)
            node_mask[i,:num_node] = 1

        assert (start_pos.sum(1) == 1).all() == True
        assert (cur_pos.sum(1) == 1).all() == True
        node_mask = node_mask.bool()
        return node_feat, edge_feat, E_matrix, E_matrix_function, query, node_mask, graph_node_all, visit_graphs, start_pos, cur_pos

    def init_graph(self, v, query, question, start_pos, h_t, policy_mapping, cur_pos):
        if policy_mapping == 'feature_only':
            v_out= self.init_node(v)
        else:
            query_emb = self.query_map(query.unsqueeze(-1))
            v_query   = torch.cat([v, query_emb], dim=-1)
            start_pos_emb       = self.start_pos_map(start_pos.unsqueeze(-1))
            v_query_start_pos   = torch.cat([v_query, start_pos_emb], dim=-1)
            if self.use_cur_pos_h:
                cur_pos_expand      = cur_pos.unsqueeze(-1).expand(cur_pos.shape + h_t.shape[-1:])
                v_query_start_pos_h = torch.cat([v_query_start_pos, \
                                                 cur_pos_expand * h_t.unsqueeze(1).expand(v.shape[0], v.shape[1], -1)], dim=-1)
            else:
                v_query_start_pos_h = torch.cat([v_query_start_pos, h_t.unsqueeze(1).expand(v.shape[0], v.shape[1], -1)], dim=-1)
            v_query_start_pos_h_q = torch.cat([v_query_start_pos_h, question.unsqueeze(1).expand(v.shape[0], v.shape[1], -1)], dim=-1)
            query_mat = query.unsqueeze(-1)
            v_out = self.init_node(v_query_start_pos_h_q) * (1 - query_mat) + self.init_query(v) * query_mat
            #v_out = self.init_node(v_query_start_pos_h_q)
        return v_out

    def init_path_selector_graph(self, v, query, question, start_pos):
        query_emb = self.query_map(query.unsqueeze(-1))
        v_query   = torch.cat([v, query_emb], dim=-1)
        start_pos_emb       = self.start_pos_map(start_pos.unsqueeze(-1))
        v_query_start_pos   = torch.cat([v_query, start_pos_emb], dim=-1)
        v_query_start_pos_q = torch.cat([v_query_start_pos, question.unsqueeze(1).expand(v.shape[0], v.shape[1], -1)], dim=-1)
        query_mat = query.unsqueeze(-1)
        v_out = self.init_path_selector_node(v_query_start_pos_q) * (1 - query_mat) + self.init_path_selector_query(v) * query_mat
        return v_out

    def get_edge_feat_planner(self, edge_feat, v_matrix, channel_idx):
        batch_size        = edge_feat.shape[0]
        edge_feat_reshape = edge_feat.transpose(2,3).transpose(1,2).contiguous().view(-1, edge_feat.shape[-3], edge_feat.shape[-2])
        edge_feat_planner = torch.zeros(batch_size, 
                                        self.pooling_num_node,
                                        self.pooling_num_node,
                                        edge_feat.shape[-1]).to(self.device)
        v_matrix_repeat   = v_matrix.unsqueeze(1).expand([batch_size, edge_feat.shape[-1]] + list(v_matrix.shape[1:]))
        v_matrix_repeat   = v_matrix_repeat.contiguous().view(-1, v_matrix.shape[1], v_matrix.shape[2]) 
        edge_feat_planner_add = torch.bmm(torch.bmm(v_matrix_repeat, edge_feat_reshape), v_matrix_repeat.transpose(1,2))
        edge_feat_planner_add = edge_feat_planner_add.view(batch_size, -1, self.pooling_num_node, self.pooling_num_node)
        edge_feat_planner    += edge_feat_planner_add.transpose(1,2).transpose(2,3)
        edge_feat_planner = self.planner_mapping[channel_idx](edge_feat_planner)
        return edge_feat_planner

    def get_uv_planner(self, v, v_matrix, channel_idx):
        batch_size = v.shape[0]
        u_planner  = torch.zeros(batch_size, self.planner_graph_dim).to(self.device)
        v_planner  = torch.zeros(batch_size, self.pooling_num_node, self.graph_dim).to(self.device)
        v_planner  = torch.bmm(v_matrix, v)
        v_planner  = self.planner_mapping[channel_idx](v_planner)
        return u_planner, v_planner
    
    def planner_forward(self, v, edge_feat, node_mask, E_matrix_function, channel_idx, num_mp_steps, ctx, ctx_mask):
        batch_size = v.shape[0]
        u_pooling  = torch.zeros(batch_size, self.pooling_graph_dim).to(self.device)
        v_pooling  = self.pooling_mapping[channel_idx](v)
        edge_feat_pooling = self.pooling_mapping[channel_idx](edge_feat)
        # graph pooling
        for t in range(self.pooling_mp_steps):
            u_pooing, v_pooling_add, _ = self.planner_pooling[channel_idx]((u_pooling, v_pooling, None), 
                                                                           (node_mask, E_matrix_function), edge_feat_pooling)
            v_pooling = v_pooling_add + v_pooling

        v_matrix  = self.pooling_mapping_2[channel_idx](v_pooling)
        v_matrix.data.masked_fill_(~node_mask.unsqueeze(-1).expand(v_matrix.shape).bool(), -float('inf'))
        v_matrix  = v_matrix.transpose(1,2)

        # normalize pooling
        if self.normalize_pooling == 'sigmoid':
            v_matrix = torch.sigmoid(v_matrix)
        elif self.normalize_pooling == 'softmax':
            v_matrix = torch.softmax(v_matrix, dim=2)

        # E_matrix preparation
        E_matrix_function_planner = torch.zeros(batch_size, 
                                                self.pooling_num_node, 
                                                self.pooling_num_node, 
                                                self.num_functions).to(self.device)
        for i in range(self.num_functions):
            E_matrix_function_planner[:,:,:,i] = torch.bmm(torch.bmm(v_matrix, E_matrix_function[:,:,:,i]), v_matrix.transpose(1,2))
        # edge_feat preparation
        edge_feat_planner = self.get_edge_feat_planner(edge_feat, v_matrix, channel_idx)
        node_mask_planner = torch.ones(batch_size, self.pooling_num_node).to(self.device)
        u_planner, v_planner = self.get_uv_planner(v, v_matrix, channel_idx)

        v_planner_L2_mean    = (v_planner * v_planner).mean()
        edge_feat_planner_L2_mean = (edge_feat_planner * edge_feat_planner).mean()

        for t in range(num_mp_steps):
            u_planner, v_planner_add, _ = self.graph_net[channel_idx]((u_planner, v_planner, None), 
                                                                      (node_mask_planner, E_matrix_function_planner), edge_feat_planner)
            v_planner = v_planner + v_planner_add
            if self.ctx_attend is not None:
                v_ctx, _ = self.ctx_attend[channel_idx](v_planner, ctx, node_mask_planner, ctx_mask)
                v_planner = self.ctx_attend_map[channel_idx](torch.cat([v_planner, v_ctx], dim=-1))

        v_residual = torch.bmm(v_matrix.transpose(1,2), v_planner)

        return v_residual, v_planner_L2_mean + edge_feat_planner_L2_mean, v_matrix

    def _get_adj_loc_list_logits(self, logits, graph_node_list_all, obs, ended):
        adj_loc_list_logits = []
        for i in range(len(graph_node_list_all)):
            adj_loc_logits = []
            if ended[i]:
                adj_loc_list_logits += [torch.zeros(len(obs[i]['adj_loc_list'])).to(self.device)]
                continue
            for loc in obs[i]['adj_loc_list']:
                vp  = 'query_' + loc['nextViewpointId']
                idx = graph_node_list_all[i].index(vp)
                adj_loc_logits += [logits[i][idx].item()]
            adj_loc_list_logits += [torch.Tensor(adj_loc_logits).to(self.device)]

        return adj_loc_list_logits

    def forward(self, obs, jump_function, model_function, model_states, action_states, \
                      batch_vp_dict, visit_graphs, num_mp_steps=2, action_query=None, \
                      question=None, top_K=10, separate_query=False, ended=None, \
                      graph_teacher_option='', is_train=False, ctx=None, ctx_mask=None, \
                      cur_step=None, window_size=None):
        """
        Use graph nets to generate a policy.
        """

        batch_size = len(batch_vp_dict)
        if self.policy_mapping == 'feature' or self.policy_mapping == 'feature_only':
            node_feat, edge_feat, E_matrix, E_matrix_function, query, node_mask, \
                graph_node_list_all, updated_visit_graphs, start_pos, cur_pos  = self._prepare_graph_feat(obs, batch_vp_dict, visit_graphs, \
                                                                                                          top_K, ended, \
                                                                                                          graph_teacher_option, is_train, \
                                                                                                          cur_step=cur_step, window_size=window_size)

            # perform graph pooling
            v = self.init_graph(node_feat, query, question, start_pos, model_states[0], self.policy_mapping, cur_pos)
            num_node = v.shape[1]
            edge_feat = self.init_edge(edge_feat)
            l2_reg_sum = 0
            if self.use_graph_pooling:
                u = torch.zeros(batch_size, v.shape[-1]).to(self.device)
                v_sum = v.clone()
                v_residual_sum = 0
                channel_list = list(range(self.pooling_channels))
                '''
                if is_train:
                    random.shuffle(channel_list)
                    num_pick = self.pooling_channels // 2
                else:
                    num_pick = self.pooling_channels
                '''
                v_matrices = []
                for i in range(self.pooling_channels):
                    #if i not in channel_list[:num_pick]:
                    #    continue
                    v_residual, l2_reg_add, v_matrix = self.planner_forward(v, edge_feat, node_mask, E_matrix_function, i, num_mp_steps, ctx, ctx_mask)
                    v_residual_sum  = v_residual_sum + self.planner_reverse_mapping[i](v_residual)
                    l2_reg_sum += l2_reg_add
                    v_matrices += [v_matrix]
                v_sum = v_sum + v_residual_sum #/ num_pick
            else:
                v_path_selector_sig = 1

                u      = torch.zeros(batch_size, v.shape[-1]).to(self.device)
                for t in range(num_mp_steps):
                    u, v_add, e = self.graph_net((u, v, None), (node_mask, E_matrix_function), edge_feat)
                    v       = v + v_add * v_path_selector_sig
                v_sum = v.clone()
                if self.policy_mapping == 'feature_only':
                    assert num_mp_steps == 0
                    v_sum = v

            _, graph_logits = self.action_attention_layer(action_query, v_sum, node_mask==0)

            graph_logits = graph_logits.unsqueeze(-1)
        elif self.policy_mapping == 'sanity_check':
            graph_logits = action_states[0]
        else:
            assert 1 == 0

        policies = []
        actions  = []
        logits   = []
        selected_nodes  = []
        entropy_total   = 0
        for i in range(batch_size):
            #logits_per = path_logits[i]
            if ended[i]:
                policies += [D.Categorical(F.softmax(torch.zeros(2).to(self.device), dim=0))]
                actions  += [-1]
                logits   += [torch.zeros(2).to(self.device)]
                selected_nodes += ['None']
                continue
            logits_per = graph_logits[i].squeeze(1)
            try:
              logits_per.data.masked_fill_((query[i] == 0).data.bool(), -float('inf'))
              probs      = F.softmax(logits_per, dim=0)
              entropy_total += self.entropy(logits_per[query[i].bool()])
            except:
              import pdb; pdb.set_trace()
            logits.append(logits_per)
            policies += [D.Categorical(probs)]
            if 'sample' in self.feedback:
                actions  += [policies[-1].sample().detach()]
            elif self.feedback == 'argmax':
                action_argmax = torch.argmax(probs)
                actions  += [action_argmax.detach()]
            selected_nodes  += [graph_node_list_all[i][actions[-1]][6:]]

        adj_loc_list_logits = self._get_adj_loc_list_logits(logits, graph_node_list_all, obs, ended)
        entropy_total = entropy_total / batch_size

        return policies, actions, logits, entropy_total, selected_nodes, \
                   graph_node_list_all, updated_visit_graphs, l2_reg_sum, adj_loc_list_logits, v_matrices
