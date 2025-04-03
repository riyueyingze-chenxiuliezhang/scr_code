import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from utils import PROJECT_DIR

with open(Path(PROJECT_DIR) / "result" / "result_reward.pkl", "rb") as f:
    result_reward = pickle.load(f)

with open(Path(PROJECT_DIR) / "result" / "result_loss.pkl", "rb") as f:
    result_loss = pickle.load(f)[2:]

with open(Path(PROJECT_DIR) / "result" / "result_conc.pkl", "rb") as f:
    result_conc = pickle.load(f)

with open(Path(PROJECT_DIR) / "result" / "result_actions.pkl", "rb") as f:
    result_actions = pickle.load(f)

print(result_actions)

plt.figure()
plt.plot(range(len(result_reward)), result_reward)
plt.title('reward')
plt.xlabel('train episode')
plt.ylabel('average reward per episode')

plt.figure()
plt.plot(range(len(result_loss)), result_loss)
plt.title('loss')
plt.xlabel('train episode')
plt.ylabel('average loss per episode')

plt.figure()
plt.plot(range(len(result_conc)), result_conc)
plt.title('outlet conc')
plt.xlabel('train episode')
plt.ylabel('average outlet conc per episode')

plt.figure()
categories = [10 + key * 2 for key in result_actions.keys()]
values = list(result_actions.values())
plt.bar(categories, values)
plt.title('action')
plt.xlabel('valve')
plt.ylabel('action times')

plt.show()
