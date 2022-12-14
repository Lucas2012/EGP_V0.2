import numpy as np
from abc import ABC

class PathSampler(ABC):
    def __init__(self, scans, graphs, batch_size, path_len_prob, name, path_len_mask=None):
        self.scans          = scans
        self.graphs         = graphs
        self.batch_size     = batch_size
        self.path_len_prob  = np.array(path_len_prob)
        self.name           = name
        self.curr_path_id   = 10000000 # Really large ID number to avoid colliding with real IDs
        #mask out certain path lengths for curriculum training
        self.path_len_mask     = np.ones(self.path_len_prob.shape) if path_len_mask is None else path_len_mask
        
    def calculate_path_len_prob(self):
        masked = self.path_len_prob * self.path_len_mask
        return masked / np.sum(masked)
    
    def reset_path_id(self):
        self.curr_path_id   = 10000000 # Really large ID number to avoid colliding with real IDs

    def construct_item(self):
        item = {}
        item['path_id']     = self.curr_path_id
        item['instr_id']    = '%s_1' %(item['path_id'])
        item['scan'],       \
        item['path'],       \
        item['prob'],       \
        item['distance']    = self.sample_path()
        item['instructions'] = "" ###### PLACEHOLDER: Will be generated by the speaker.
        item['heading']     = 0 
        self.curr_path_id  += 1
        return item 
    
    def sample_batch(self): 
        batch = []
        for i in range(self.batch_size):
            batch.append(self.construct_item())
        return batch
    
    def _sample_index(self, prob_vector):
        index = np.random.choice(len(prob_vector), 1, p=prob_vector)[0]
        return index, prob_vector[index]
    
    def sample_path(self):
        pass
    
    
class RandomPathSampler(PathSampler):
    def __init__(self, scans, graphs, batch_size, path_len_prob, name='RandomPathSampler', distances=None, path_len_mask=None, norepeat=False, min_distance=0):
        super(RandomPathSampler, self).__init__(scans, graphs, batch_size, path_len_prob, name, path_len_mask)
        self._calculate_scan_sample_prob()
        self.norepeat = norepeat
        self.min_distance = min_distance
        self.distances = distances
        self.count_threshold = 100
                
    def _calculate_scan_sample_prob(self):
        # For all the scans in the split, we calculate prob of sampling the scan based on the number
        # of nodes in the scan. This allows us to uniformly sample starting position across nodes.
        scan_sample_prob      = []
        for scan in self.scans:
            scan_sample_prob.append(len(self.graphs[scan].nodes()))
        scan_sample_prob      = np.array(scan_sample_prob)
        self.scan_sample_prob = scan_sample_prob / np.sum(scan_sample_prob)
        
    def sample_path(self):
        init_path_prob = 1
        distance  = 0
        
        # Sample path length
        path_len_prob      = self.calculate_path_len_prob()
        path_len_ind, prob = self._sample_index(path_len_prob)
        path_len           = path_len_ind + 1 #The path length is just the index sampled + 1. Path length is defined by num edge, not num node.
        init_path_prob         *= prob
        
        # Sample scan and update probability
        scan_ind, prob = self._sample_index(self.scan_sample_prob)
        scan           = self.scans[scan_ind]
        init_path_prob     *= prob

        count = 0
        while True:        
            # starting path_prob
            path_prob = init_path_prob * 1.0
            # Sample from graph
            path  = []
            graph = self.graphs[scan]
            # Sample initial node
            init_node_ind, \
            prob             = self._sample_index(np.full(len(graph.nodes()), 1/len(graph.nodes())))
            node             = list(graph.nodes)[init_node_ind]
            path.append(node)
            path_prob       *= prob
            # Sample the rest of the path
            for i in range(path_len):
                adj_nodes       = graph.adj[node]
                adj_nodes_list  = list(adj_nodes)
                if self.norepeat:
                    for ele in path:
                        if ele in adj_nodes_list and len(adj_nodes_list) > 1:
                            adj_nodes_list.remove(ele)
                adj_node_ind, \
                prob            = self._sample_index(np.full(len(adj_nodes_list), 1/len(adj_nodes_list)))
                #adj_node        = list(adj_nodes)[adj_node_ind]
                adj_node        = adj_nodes_list[adj_node_ind]
                distance       += adj_nodes[adj_node]['weight']
                path_prob      *= prob
                node = adj_node
                path.append(adj_node)
            # distance threshold
            if self.distances[scan][path[0]][path[-1]] > self.min_distance:
                break
            count += 1
            if count > self.count_threshold:
                print('Sampled too many times (>100), use the path as it is now.')
                print('Distance: ', self.distances[scan][path[0]][path[-1]])
                break
            
        return scan, path, path_prob, distance

    
class ShortestPathSampler(PathSampler):
    def __init__(self, scans, graphs, batch_size, path_len_prob, name='ShortestPathSampler', paths=None, distances=None, path_len_mask=None):
        super(ShortestPathSampler, self).__init__(scans, graphs, batch_size, path_len_prob, name, path_len_mask)
        if paths is None or distances is None:
            raise ValueError('Please pass in a keyword argument for paths and distances')
        self._format_paths(paths)
        self.distances = distances
        
    def _format_paths(self, paths):
        path_len_count  = {} # dict[path len, list (# paths per scan of len, order corresponds with self.scans)]
        formatted_paths = {} # dict[scan, dict[path len, paths in scan of len]]
        
        # Loop through scans -> start node -> end node to get all paths in all scans
        for scan_ind in range(len(self.scans)):
            scan          = self.scans[scan_ind]
            paths_in_scan = paths[scan]
            start_nodes   = list(paths_in_scan.keys())
            
            for start_node in start_nodes:
                paths_from_node = paths_in_scan[start_node]
                end_nodes       = list(paths_from_node.keys())
                
                for end_node in end_nodes:
                    # get path and length
                    path     = paths_from_node[end_node]
                    path_len = len(path)-1
                    # if path is too short or long, we ignore it
                    if path_len < 1 or path_len > len(self.path_len_prob): continue
                    # update our two dictionary structures
                    count                    = path_len_count.setdefault(path_len, np.zeros(len(self.scans)))
                    count[scan_ind]         += 1
                    temp_paths_dict      = formatted_paths.setdefault(scan, {})
                    paths_in_scan_of_len = temp_paths_dict.setdefault(path_len, list()).append(path)
                    
        self.path_len_count = path_len_count
        self.formatted_paths = formatted_paths
        
    # Samples a path by first sampling a path length with probability defined by path_len_prob and mask.
    # Afterward, sample a path with that length with uniform probability across all shortest paths of that length.
    def sample_path(self):
        path_prob = 1
        distance = 0
        
        # Sample path length
        path_len_prob   = self.calculate_path_len_prob()
        path_len_ind, \
        prob            = self._sample_index(path_len_prob)
        path_len        = path_len_ind + 1 #The path length is just the index sampled + 1. 
        path_prob      *= prob
        
        # Sample scan and update probability
        scan_probs  = self.path_len_count[path_len] / np.sum(self.path_len_count[path_len])
        scan_ind, \
        prob        = self._sample_index(scan_probs)
        scan        = self.scans[scan_ind]
        path_prob  *= prob
        
        # Sample path uniformly from scan and path length
        possible_paths = self.formatted_paths[scan][path_len]
        path_ind, \
        prob           = self._sample_index(np.full(len(possible_paths), 1/len(possible_paths)))
        path           = possible_paths[path_ind]
        distance       = self.distances[scan][path[0]][path[-1]]
        path_prob     *= prob
        
        return scan, path, path_prob, distance
