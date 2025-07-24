import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# plt.plot(env.episode_history, env.reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Episode")
plt.title("RL Training (DQN): Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("k.png")
