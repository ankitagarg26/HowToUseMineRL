from collections import deque
import numpy as np

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path):
        b = np.load(path+'.npy', allow_pickle=True)
#         assert(b.shape[0] == self.memory_size)

        for i in range(b.shape[0]):
            self.add(b[i])