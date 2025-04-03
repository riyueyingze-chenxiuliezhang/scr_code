import random
from pathlib import Path

import torch

from .dqn_impl import DQNImpl, DQNAgentImpl
from project.utils.tool import ReplayMemory


class DQN(DQNImpl):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)


class DQNAgent(DQNAgentImpl):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(config)

        self._dqn = DQN(state_dim, action_dim, config)

        self._action_dim = action_dim
        self._config = config

        if config.name == "train":
            self._start_size = config.start_size
            self._batch_size = config.batch_size
            self._replay_memory = ReplayMemory(config.capacity)

    def select_action(self, state):
        if random.uniform(0, 1) < self._dqn.epsilon:
            action = random.randrange(0, self._action_dim)
            action_value = None
        else:
            state = torch.from_numpy(state.reshape(1, -1)).type(torch.float32).to(self._dqn.device)
            action_values = self._dqn.get_q_value(state).max(1)
            action_value = action_values[0].item()
            action = action_values[1].item()
        return action, action_value

    def update_network(self):
        if len(self._replay_memory) < self._start_size:
            return None

        s, a, r, ns, d = self._replay_memory.sample(self._batch_size)
        transition_dict = {
            "states": s,
            "actions": a,
            "next_states": ns,
            "rewards": r,
            "done": d
        }
        loss = self._dqn.train(transition_dict)
        return loss.item()

    def add_experience(self, state, action, reward, next_state, done):
        self._replay_memory.add(state, action, reward, next_state, done)

    def save_network(self, dir_path: Path, epoch):
        dir_path.mkdir(exist_ok=True, parents=True)
        self._dqn.save(dir_path / f"checkpoint_{epoch}.pth")

    def load_network(self, dir_path: Path, _epoch):
        self._dqn.load(dir_path / f"checkpoint_{_epoch}.pth")


if __name__ == '__main__':
    pass
