import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Define SUMO configuration
Sumo_config = [
    'sumo',
    '-c', 'simulation_run_rl.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 4: Define Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)  # 0 = keep phase, 1 = switch phase
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)  # (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)
        self.min_green_steps = 100
        self.step_count = 0
        self.max_steps = 1000  # Steps per episode
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0

        # Lists to record data for plotting
        self.episode_history = []
        self.reward_history = []
        self.queue_history = []

    def reset(self, seed=None, **kwargs):
        # Close any existing SUMO connection
        if traci.isLoaded():
            traci.close()
        # Start new SUMO simulation
        traci.start(self.config)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        state = self._get_state()
        info = {"episode": len(self.episode_history) + 1}
        return state, info

    def step(self, action):
        self.current_simulation_step = self.step_count
        self._apply_action(action)
        traci.simulationStep()
        new_state = self._get_state()
        reward = self._get_reward(new_state)
        self.cumulative_reward += reward
        self.total_queue += sum(new_state[:-1])
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        info = {}
        if done:
            avg_queue = self.total_queue / self.step_count
            self.episode_history.append(len(self.episode_history))
            self.reward_history.append(self.cumulative_reward)
            self.queue_history.append(avg_queue)
            info = {
                "episode": len(self.episode_history),
                "cumulative_reward": self.cumulative_reward,
                "avg_queue_length": avg_queue
            }
            print(f"Episode {info['episode']} Summary: Cumulative Reward: {info['cumulative_reward']:.2f}, Avg Queue Length: {info['avg_queue_length']:.2f}")

        return new_state, reward, terminated, truncated, info

    def _get_state(self):
        detector_Node1_2_EB_0 = "Node1_2_EB_0"
        detector_Node1_2_EB_1 = "Node1_2_EB_1"
        detector_Node1_2_EB_2 = "Node1_2_EB_2"
        detector_Node2_7_SB_0 = "Node2_7_SB_0"
        detector_Node2_7_SB_1 = "Node2_7_SB_1"
        detector_Node2_7_SB_2 = "Node2_7_SB_2"
        traffic_light_id = "Node2"

        q_EB_0 = self._get_queue_length(detector_Node1_2_EB_0)
        q_EB_1 = self._get_queue_length(detector_Node1_2_EB_1)
        q_EB_2 = self._get_queue_length(detector_Node1_2_EB_2)
        q_SB_0 = self._get_queue_length(detector_Node2_7_SB_0)
        q_SB_1 = self._get_queue_length(detector_Node2_7_SB_1)
        q_SB_2 = self._get_queue_length(detector_Node2_7_SB_2)
        current_phase = self._get_current_phase(traffic_light_id)

        return np.array([q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase], dtype=np.float32)

    def _apply_action(self, action, tls_id="Node2"):
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                next_phase = (self._get_current_phase(tls_id) + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
                self.last_switch_step = self.current_simulation_step

    def _get_reward(self, state):
        total_queue = sum(state[:-1])  # Exclude current_phase
        reward = -float(total_queue)
        return reward

    def _get_queue_length(self, detector_id):
        return traci.lanearea.getLastStepVehicleNumber(detector_id)

    def _get_current_phase(self, tls_id):
        return traci.trafficlight.getPhase(tls_id)

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode="human"):
        pass  # No rendering for non-GUI SUMO

# Step 5: Episode-based Training Loop with Stable Baselines3
print("\n=== Starting Episode-based Reinforcement Learning (DQN with Stable Baselines3) ===")

# Initialize environment
env = SumoEnv(Sumo_config)

# Check environment compatibility
from stable_baselines3.common.env_checker import check_env
check_env(env)

# Initialize DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.1,  # ALPHA
    gamma=0.9,          # GAMMA
    exploration_initial_eps=0.1,  # EPSILON
    exploration_final_eps=0.1,    # Keep constant for consistency
    exploration_fraction=1.0,     # Apply epsilon for entire training
    verbose=1,
    learning_starts=0,  # Start learning immediately
    train_freq=1,       # Train after every step
    batch_size=32,
    target_update_interval=1000
)

# Define callback for logging
class CustomEvalCallback(EvalCallback):
    def __init__(self, env, **kwargs):
        super().__init__(eval_env=env, **kwargs)
        self.env = env

    def _on_step(self):
        super()._on_step()
        return True

# Train for 100 episodes of 1000 steps each
eval_callback = CustomEvalCallback(
    env,
    eval_freq=1000,  # Evaluate after each episode
    n_eval_episodes=1,
    deterministic=False,
    verbose=1
)

total_timesteps = 100 * 1000  # 100 episodes * 1000 steps
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# Save the model
model.save("dqn_sumo")

# Close the environment
env.close()

# Step 6: Visualization of Results
plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.queue_history, marker='o', linestyle='-', label="Average Queue Length")
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("RL Training (DQN): Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.show()