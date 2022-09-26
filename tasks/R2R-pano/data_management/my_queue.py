'''
  My implementation of queue
'''

class MyQueue(object):
    """
    A queue class.
    """

    def __init__(self):
        self.queue = []

    def add(self, entry):
        self.queue.append(entry)

    def get_by_idx(self, index):
        assert(index < self.size())
        return self.queue[index]

    def get_by_key(self, key):
        return [ele.node_dict[key] for ele in self.queue]

    def get_by_key_active(self, key):
        if key == 'index':
            results = [i for i, ele in enumerate(self.queue) if ele.node_dict['active']]
        else:
            results = [ele.node_dict[key] for ele in self.queue if ele.node_dict['active']]
        return results

    def get(self, act, key):
        return self.queue[act].node_dict[key]

    def deactivate(self, act):
        self.queue[act].node_dict['active'] = False

    def size(self):
        return len(self.queue)

    def size_active(self):
        count = 0
        for ele in self.queue:
            if ele.node_dict['active']:
                count += 1
        return count
