"""
# @Time: 2025/3/26 14:07
# @File: train.py
"""
from tqdm import tqdm

from utils import TrainSCREnv, DQNAgent, DataRecorder
from config import args, ConfigLoader, MODEL_SAVE_DIR, STAT_SAVE_DIR

env = TrainSCREnv()
action_dim = env.action_dim
state_dim = env.state_dim
agent = DQNAgent(state_dim, action_dim, ConfigLoader.dqn.train)


def train(episode):
    """ 真实数据训练，action由数据提供 """
    stat_path_episode = STAT_SAVE_DIR / str(episode)
    stat_path_episode.mkdir(exist_ok=True, parents=True)
    data_record = DataRecorder(stat_path_episode, file_fmt="pkl")

    env.reset()
    epoch = 0
    process_done = False
    progress_bar = tqdm(total=env.data_num, desc=f"Ep {episode} / {args.epochs - 1} 训练")

    while not process_done:
        epoch += 1
        episode_loss = []
        episode_reward = []

        for _ in range(300):
            state, action, reward, next_state, done, process_done = env.step()
            agent.add_experience(state, action, reward, next_state, done)

            loss = agent.update_network()
            if loss:
                episode_loss.append(loss)
            episode_reward.append(reward)

            progress_bar.update(1)

            if process_done:
                break

        mean_reward = sum(episode_reward) / len(episode_reward)
        data_record.add_data("result_reward", mean_reward)

        mean_loss = sum(episode_loss) / len(episode_loss) if len(episode_loss) != 0 else 1
        data_record.add_data("result_loss", mean_loss)

        print(f"mean_reward: {mean_reward}   "
              f"mean_loos: {mean_loss}")

    progress_bar.close()

    agent.save_network(MODEL_SAVE_DIR, episode)
    data_record.flush()


if __name__ == '__main__':
    for e in range(args.epochs):
        train(e)
