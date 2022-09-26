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

from utils import padding_idx, end_token_idx, pad_list_tensors, kl_div, CrossEntropy
from data_management import MyQueue, NodeState

import pdb



class PanoBaseAgent(object):
    """ Base class for an R2R agent with panoramic view and action. """

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = OrderedDict()
    
    def write_results(self):
        output = []
        for k, v in self.results.items():
            output.append(
                OrderedDict([
                    ('instr_id', k),
                    ('trajectory', v['path']),
                ])
            )
        with open(self.results_path, 'w') as f:
            json.dump(output, f)
    
    def _get_distance(self, ob):
        try:
            gt = self.gt[int(ob['instr_id'].split('_')[0])]
        except:  # synthetic data only has 1 instruction per path
            gt = self.gt[int(ob['instr_id'])]
        distance = self.env.distances[ob['scan']][ob['viewpoint']][gt['path'][-1]]
        return distance

    def _select_action(self, logit, ended, is_prob=False, fix_action_ended=True):
        logit_cpu = logit.clone().cpu()
        if is_prob:
            probs = logit_cpu
        else:
            probs = F.softmax(logit_cpu, 1)

        if self.feedback == 'argmax':
            _, action = probs.max(1)  # student forcing - argmax
            action = action.detach()
        elif 'sample' in self.feedback:
            # sampling an action from model
            m = D.Categorical(probs)
            action = m.sample()
        else:
            raise ValueError('Invalid feedback option: {}'.format(self.feedback))

        # set action to 0 if already ended
        if fix_action_ended:
            for i, _ended in enumerate(ended):
                if _ended:
                    action[i] = self.ignore_index

        return action

    def _next_viewpoint(self, obs, viewpoints, navigable_index, action, ended):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] >= 1 and not action[i] == self.ignore_index:
                next_viewpoint_idx.append(navigable_index[i][action[i] - 1])  # -1 because the first one in action is 'stop'
            else:
                next_viewpoint_idx.append('STAY')
                ended[i] = True

            # use the available viewpoints and action to select next viewpoint
            next_action = 0 if action[i] == self.ignore_index else action[i]
            next_viewpoints.append(viewpoints[i][next_action])
            # obtain the heading associated with next viewpoints
            next_headings.append(ob['navigableLocations'][next_viewpoints[i]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def pano_navigable_feat(self, obs, ended, next_path_idx=None):

        # Get the 36 image features for the panoramic view (including top, middle, bottom)
        num_feature, feature_size = obs[0]['feature'].shape

        pano_img_feat = torch.zeros(len(obs), num_feature, feature_size)
        navigable_feat = torch.zeros(len(obs), self.opts.max_navigable, feature_size)

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            pano_img_feat[i, :] = torch.from_numpy(ob['feature'])  # pano feature: (batchsize, 36 directions, 2048)

            index_list = []
            viewpoints_tmp = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']
            # If we want to follow the GT traj exactly, we can't rely on the targets found above.
            if next_path_idx is not None:
                goal_viewpoint = ob['teacher'][next_path_idx[i]]
                # If we have reached the current objective but haven't reached the end of the path,
                # increment path_idx to set next step as objective.
                if ob['viewpoint'] == goal_viewpoint and next_path_idx[i] < len(ob['teacher'])-1:
                    next_path_idx[i] += 1
                    goal_viewpoint = ob['teacher'][next_path_idx[i]]
                teacher_path = self.env.paths[ob['scan']][ob['viewpoint']][goal_viewpoint]
                if len(teacher_path) > 1:
                    gt_viewpoint_id = teacher_path[1]
                else:
                    # Due to the previous if statement, this is only possible if the current viewpoint
                    # has reached the end of the entire path.
                    gt_viewpoint_id = ob['viewpoint']
            

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)

            navigable_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]

        return pano_img_feat, navigable_feat, (viewpoints, navigable_feat_index, target_index)

    def teacher_forcing_target(self, step, obs, ended):
        target_index = []
        for i, ob in enumerate(obs):
            gt_viewpoint_id = ob['teacher'][min(step+1, len(ob['teacher'])-1)]
            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                if viewpoint_id == gt_viewpoint_id:
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)
        return target_index 

    def _sort_batch(self, obs, only_keep_five=False):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        if only_keep_five:
            num_keep = 5
            for i, row in enumerate(seq_tensor):
                seq_length = seq_lengths[i]
                if seq_length <= num_keep + 2: # One extra for start, and one for end       
                    continue
                if seq_length == seq_tensor.shape[1]: # Rare edge case, but need to add end token
                    seq_tensor[i,1:num_keep+1] = row[seq_length-num_keep:seq_length]
                    seq_tensor[i, num_keep+1]  = end_token_idx 
                else:
                    seq_tensor[i,1:7] = row[seq_length-num_keep-1:seq_length]
                seq_tensor[i, 7:] = padding_idx
                seq_lengths[i] = num_keep + 2
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)

    def _sort_batch_from_seq(self, seq_list):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array(seq_list)
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)
