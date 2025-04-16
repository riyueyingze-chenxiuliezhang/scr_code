"""
# @Time: 2025/3/26 14:07
# @File: test.py
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import TestSCREnv, DQNAgent, DataRecorder
from config import ConfigLoader, PROJECT_DIR, MODEL_SAVE_DIR

env = TestSCREnv()
action_dim = env.action_dim
state_dim = env.state_dim
agent = DQNAgent(state_dim, action_dim, ConfigLoader.dqn.test)

action_config = ConfigLoader.environment.action
actions = np.arange(action_config.action_min,
                    action_config.action_max + action_config.action_step,
                    action_config.action_step)


def test(episode):
    """ 真实数据训练，action由数据提供 """
    agent.load_network(MODEL_SAVE_DIR / f"checkpoint_{episode}.pth")
    test_result_path = PROJECT_DIR / "result" / "test" / ConfigLoader.name / str(episode)
    test_result_path.mkdir(exist_ok=True, parents=True)

    data_recode = DataRecorder(test_result_path, file_fmt="pkl")

    state, information = env.reset()

    done = False
    progress_bar = tqdm(total=env.data_num, desc=f"Ep {episode} 测试")

    while not done:
        for _ in range(300):
            action_index, action_value = agent.select_action(state)
            action = actions[action_index]
            raw_data_row = information['raw_data_row']

            time = pd.to_datetime(raw_data_row['时间'], format="%Y/%m/%d %H:%M:%S", errors="raise")
            real_action = raw_data_row['焦炉煤气阀门开度']
            real_target = raw_data_row['目标浓度']
            real_outlet_c = raw_data_row['出口NO2浓度（折算）']

            data_recode.add_data("time", time.strftime("%Y/%m/%d %H:%M:%S"))
            data_recode.add_data("action", int(action))
            data_recode.add_data("action_value", action_value)
            data_recode.add_data("real_action", int(real_action))
            data_recode.add_data("real_target", int(real_target))
            data_recode.add_data("real_outlet_c", real_outlet_c)

            state, _, done, information = env.step()

            progress_bar.update(1)

            if done:
                break

    progress_bar.close()
    data_recode.flush()


if __name__ == '__main__':
    for e in range(0, 20):
        test(e)
