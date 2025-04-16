"""
# @Time: 2025/3/26 14:07
# @File: train.py
"""
from utils import SCREnv, DQNAgent
from config import ConfigLoader

env = SCREnv()
action_dim = env.action_dim
state_dim = env.state_dim
agent = DQNAgent(action_dim, state_dim, ConfigLoader.dqn.train)
