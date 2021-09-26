import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, his_t_c, action, reward, his_tp1, done):
        data = (his_t_c, action, reward, his_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        hises_t_c, actions, rewards, hises_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            his_t_c, action, reward, his_tp1, done = data
            hises_t_c.append(np.array(his_t_c, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            hises_tp1.append(np.array(his_tp1, copy=False))
            dones.append(done)
        return np.array(hises_t_c), np.array(actions), np.array(rewards), np.array(hises_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        his_batch: np.array
            batch of history trajectory
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_his_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
