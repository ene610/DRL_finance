import heapq
from itertools import count
tiebreaker = count()
import numpy as np

class ReplayBuffer(object):
    def __init__(self, input_shape, n_actions, seed, max_size = 10000):
        # self.mem_size = max_size
        # self.mem_cntr = 0
        # self.seed = seed
        # self.state_memory = np.zeros((self.mem_size, *input_shape),
        #                              dtype=np.float32)
        # self.new_state_memory = np.zeros((self.mem_size, *input_shape),
        #                                  dtype=np.float32)
        #
        # self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        # self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        # self.seed = seed
        # self.rng = np.random.default_rng(self.seed)

        self.max_size = max_size
        self.memory = []

    def store_transition(self, state, action, reward, state_, done, TDerror):
        transition = [state,action,reward,state_,done]
        heapq.heappush(self.memory, (-TDerror, next(tiebreaker), transition))
        if self.size() > self.max_size:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)

    def sample_buffer(self, batch_size):

        batch = heapq.nsmallest(batch_size, self.memory)
        batch = [e for (_, _, e) in batch]

        self.memory = self.memory[batch_size:]

        transition = np.array(batch)
        transition = transition.transpose()

        return transition[0], transition[1], transition[2], transition[3], transition[4]



class PER():
    """ Prioritized replay memory using binary heap """

    def __init__(self, max_size=3):
        self.max_size = max_size
        self.memory = []

    def add(self, transition, TDerror):
        heapq.heappush(self.memory, (-TDerror, next(tiebreaker), transition))
        if self.size() > self.max_size:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)

    def batch(self, n):
        batch = heapq.nsmallest(n, self.memory)
        batch = [e for (_, _, e) in batch]
        self.memory = self.memory[n:]
        ciao = np.array(batch)
        ciao = ciao.transpose()
        print("batch",batch)
        print("ciao",ciao)
        return ciao[0],ciao[1],ciao[2],ciao[3],ciao[4]

    def size(self):
        return len(self.memory)

    def is_full(self):
        return True if self.size() >= self.max_size else False

alfa = PER()
alfa.add([2, 1, 3, 4, 5], 10)
alfa.add([2, 1, 4, 4, 5], 20)
alfa.add([1, 1, 3, 4, 5], 66)
alfa.add([1, 1, 3, 4, 5], 10)
alfa.add([1, 1, 3, 4, 5], 10)
alfa.add([1, 1, 3, 4, 5], 10)
alfa.add([1, 1, 3, 4, 5], 10)

print(alfa.memory)
print(alfa.batch(3))
print(alfa.memory)
