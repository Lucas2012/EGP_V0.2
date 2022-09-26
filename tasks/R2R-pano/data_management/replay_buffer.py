import numpy as np
import random

class ReplayBuffer(object):
    """
    A class used to save and replay data.
    """

    def __init__(self, buffer_size=1000):
        self.re_buffer   = []
        self.buffer_size = buffer_size

    def add_path(self, static_ctx_features, path_vp_dict, path_tuples):
        new_path = {}
        new_path['static_ctx_features'] = static_ctx_features
        new_path['path_vp_dict'] = path_vp_dict
        new_path['path_tuples']  = path_tuples
        if self.buffer_size <= len(self.re_buffer):
            del self.re_buffer[0]
        self.re_buffer.append(new_path)

    def random_batch(self, batch_size=64):
        buffer_len = len(self.re_buffer)
        selected_items = list(range(buffer_len))
        random.shuffle(selected_items)
        selected_items = selected_items[:batch_size]

        batch_static_ctx_features = []
        batch_vp_dict  = []
        batch_state    = []
        batch_reward   = []
        batch_state_prime = []
        batch_done     = []
        for ele in selected_items:
            batch_static_ctx_features.append(self.re_buffer[ele]['static_ctx_features'])
            batch_vp_dict.append(self.re_buffer[ele]['path_vp_dict'])
            tuple_index = list(range(len(self.re_buffer[ele]['path_tuples'])))
            random.shuffle(tuple_index)
            tuple_index = tuple_index[0]
            tuple_ele   = self.re_buffer[ele]['path_tuples'][tuple_index]
            batch_state.append(tuple_ele[:4])
            batch_reward.append(tuple_ele[4])
            batch_state_prime.append(tuple_ele[5:9])
            batch_done.append(tuple_ele[-1])

        return batch_static_ctx_features, batch_vp_dict, batch_state, batch_reward, batch_state_prime, batch_done
