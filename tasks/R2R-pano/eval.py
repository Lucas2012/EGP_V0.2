''' Evaluation of agent trajectories '''

import json
from collections import defaultdict
from collections import OrderedDict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs
from cls import CLS
from dtw import DTW


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, dataset_name, online=False, non_shortest_path=False):
        self.error_margin = 3.0
        self.splits = splits
        self.dataset_name = dataset_name
        self.gt = OrderedDict()
        self.instr_ids = []
        self.scans = []
        self.online = online
        self.non_shortest_path = non_shortest_path

        if not self.online:
            for item in load_datasets(splits, dataset_name=self.dataset_name):
                self.gt[item['path_id']] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
            self._process()

        self.cls = CLS(graphs=self.graphs)
        self.dtw = DTW(graphs=self.graphs)

    #def _load_datasets_online(self, filename):
    #    for item in load_datasets(splits=None, filename=filename):
    #        self.gt[item['path_id']] = item
    #        self.scans.append(item['scan'])
    #        self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
    #    self._process()

    def _clear_data(self):
        self.gt = OrderedDict()
        self.instr_ids = []
        self.scans = []

    def _process(self):
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = OrderedDict()
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)

        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        is_success = self.distances[gt['scan']][final_position][goal] < self.error_margin

        gt_distance = 0  # Work out the length of the path in meters
        gt_prev     = gt['path'][0]
        for gt_curr in gt['path'][1:]:
            gt_distance += self.distances[gt['scan']][gt_prev][gt_curr]
            gt_prev = gt_curr

        if self.dataset_name == 'R4R':
            gt_distance = gt_distance
        else:
            gt_distance = self.distances[gt['scan']][start][goal]
        self.scores['success_path_length'].append(
            is_success * self.distances[gt['scan']][start][goal] / max(gt_distance, distance))

        generated_path = [ele[0] for ele in path]

        self.scores['cls'].append(self.cls(gt['scan'], generated_path, gt['path']))
        self.scores['sdtw'].append(self.dtw(gt['scan'], generated_path, gt['path'], metric='sdtw'))
        self.scores['ndtw'].append(self.dtw(gt['scan'], generated_path, gt['path'], metric='ndtw'))

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids) 
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = OrderedDict([
            ('nav_error', np.average(self.scores['nav_errors'])),
            ('oracle_error', np.average(self.scores['oracle_errors'])),
            ('steps', np.average(self.scores['trajectory_steps'])),
            ('lengths', np.average(self.scores['trajectory_lengths'])),
            ('spl', np.average(self.scores['success_path_length'])),
            ('cls', np.average(self.scores['cls'])),
            ('sdtw', np.average(self.scores['sdtw'])),
            ('ndtw', np.average(self.scores['ndtw'])),
        ])
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        return score_summary, self.scores

    def get_dtw(self, scan, prediction, reference):
        assert 1 == 0
        margin = 3.0
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                best_previous_cost = min(
                    dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                cost = self.distances[scan][prediction[i - 1]][reference[j - 1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        ndtw = np.exp(-dtw / (margin * len(prediction)))
        return ndtw

    def get_cls(self, scan, trajectory, path):
        assert 1 == 0
        decay = 3
        pc = 0
        path_pl = 0
        traj_pl = 0
        for i, loc in enumerate(path):
            if i < len(path) - 1:
                path_pl += self.distances[scan][path[i]][path[i + 1]]
            nearest = np.inf
            for pred_loc in trajectory:
                if self.distances[scan][loc][pred_loc] < nearest:
                    nearest = self.distances[scan][loc][pred_loc]
            pc += np.exp(-nearest / decay)
        pc /= len(path)
        epl = pc * path_pl
        for i in range(len(trajectory) - 1):
            traj_pl += self.distances[scan][trajectory[i]][trajectory[i + 1]]
        if epl == 0 and traj_pl == 0:
            cls = 0
        else:
            cls = epl / (epl + abs(epl - traj_pl))
        return cls


    def _score_item_online(self, gt, path, scores):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)

        scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        scores['trajectory_steps'].append(len(path)-1)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        scores['trajectory_lengths'].append(distance)
        is_success = self.distances[gt['scan']][final_position][goal] < self.error_margin

        gt_distance = 0  # Work out the length of the path in meters
        gt_prev     = gt['path'][0]
        for gt_curr in gt['path'][1:]:
            gt_distance += self.distances[gt['scan']][gt_prev][gt_curr]
            gt_prev = gt_curr

        if self.dataset_name == 'R4R':
            gt_distance = gt_distance
        else:
            gt_distance = self.distances[gt['scan']][start][goal]
        scores['success_path_length'].append(is_success * self.distances[gt['scan']][start][goal] / max(gt_distance, distance))

        generated_path = [ele[0] for ele in path]

        scores['cls'] = self.cls(gt['scan'], generated_path, gt['path'])
        scores['sdtw'] = self.dtw(gt['scan'], generated_path, gt['path'], metric='sdtw')
        scores['ndtw'] = self.dtw(gt['scan'], generated_path, gt['path'], metric='ndtw')

        return scores

    def score_online(self, output_file, gt):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        scores = defaultdict(list)
        path_ids = [gt[k]['path_id'] for k in gt.keys()]
        path_ids = set(path_ids) 
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['path_id'] in path_ids:
                    path_ids.remove(item['path_id'])
                    scores = self._score_item_online(gt[item['path_id']], item['trajectory'], scores)
      
        score_summary = OrderedDict([
            ('nav_error', np.average(scores['nav_errors'])),
            ('oracle_error', np.average(scores['oracle_errors'])),
            ('steps', np.average(scores['trajectory_steps'])),
            ('lengths', np.average(scores['trajectory_lengths'])),
            ('spl', np.average(scores['success_path_length'])),
            ('cls', np.average(scores['cls'])),
            ('sdtw', np.average(scores['sdtw'])),
            ('ndtw', np.average(scores['ndtw'])),
        ])
        num_successes = len([i for i in scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(scores['nav_errors']))
        oracle_successes = len([i for i in scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(scores['oracle_errors']))

        return score_summary, scores
