import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
Sumo_config = [
    'sumo',  # Use 'sumo-gui' temporarily for debugging to visualize the simulation
    '-c', 'simulation_run_rl.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)  # 0 = keep phase, 1 = switch phase
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
        self.min_green_steps = 100
        self.step_count = 0
        self.max_steps = 1000
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        self.episode_count = 0
        self.episode_history = []
        self.reward_history = []
        self.queue_history = []

    def reset(self, seed=None, **kwargs):
        # Close any existing SUMO connection
        if traci.isLoaded():
            traci.close()
        # Start new SUMO simulation with error handling
        try:
            traci.start(self.config)
        except traci.TraCIException as e:
            sys.exit(f"Failed to start SUMO: {e}")
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        self.episode_count += 1
        state = self._get_state()
        if not isinstance(state, np.ndarray) or state.shape != (7,):
            raise ValueError(f"Invalid state: {state}")
        return state, {"episode": self.episode_count}

    def step(self, action):
        self.current_simulation_step = self.step_count
        self._apply_action(action)
        try:
            traci.simulationStep()
        except traci.TraCIException as e:
            print(f"TraCI error during step: {e}")
            return self._get_state(), 0.0, True, False, {"error": "TraCI failure"}
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
            avg_queue = self.total_queue / self.step_count if self.step_count > 0 else 0
            self.episode_history.append(self.episode_count - 1)
            self.reward_history.append(self.cumulative_reward)
            self.queue_history.append(avg_queue)
            info = {
                "episode": self.episode_count,
                "cumulative_reward": self.cumulative_reward,
                "avg_queue_length": avg_queue
            }
            print(f"Episode {info['episode']} Summary: Cumulative Reward: {info['cumulative_reward']:.2f}, "
                  f"Avg Queue Length: {info['avg_queue_length']:.2f}, Last State: {new_state}, Reward: {reward}")

        return new_state, reward, terminated, truncated, info

    def _get_state(self):
        detector_ids = [
            "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
            "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
        ]
        traffic_light_id = "Node2"

        queue_lengths = []
        for detector_id in detector_ids:
            try:
                q = traci.lanearea.getLastStepVehicleNumber(detector_id)
                if not isinstance(q, (int, float)) or q < 0:
                    print(f"Invalid queue length for {detector_id}: {q}")
                    q = 0.0
                queue_lengths.append(float(q))
            except traci.TraCIException as e:
                print(f"Error getting queue length for {detector_id}: {e}")
                queue_lengths.append(0.0)
        
        try:
            current_phase = float(self._get_current_phase(traffic_light_id))
        except traci.TraCIException as e:
            print(f"Error getting current phase: {e}")
            current_phase = 0.0

        state = np.array(queue_lengths + [current_phase], dtype=np.float32)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Invalid state values: {state}")
            state = np.zeros(7, dtype=np.float32)  # Fallback to zero state
        return state

    def _apply_action(self, action, tls_id="Node2"):
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                try:
                    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                    num_phases = len(program.phases)
                    next_phase = (self._get_current_phase(tls_id) + 1) % num_phases
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                except traci.TraCIException as e:
                    print(f"Error applying action: {e}")

    def _get_reward(self, state):
        total_queue = sum(state[:-1])  # Exclude current_phase
        if np.isnan(total_queue) or np.isinf(total_queue):
            print(f"Invalid total_queue: {total_queue}, State: {state}")
            total_queue = 0.0
        reward = float(-total_queue)  # Ensure Python float
        if not isinstance(reward, float):
            print(f"Reward is not a float: {reward}, type: {type(reward)}")
            reward = 0.0
        return reward

    def _get_queue_length(self, detector_id):
        try:
            return traci.lanearea.getLastStepVehicleNumber(detector_id)
        except traci.TraCIException as e:
            print(f"Error in get_queue_length for {detector_id}: {e}")
            return 0.0

    def _get_current_phase(self, tls_id):
        try:
            return traci.trafficlight.getPhase(tls_id)
        except traci.TraCIException as e:
            print(f"Error in get_current_phase: {e}")
            return 0

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode="human"):
        pass

# Custom Callback
class EpisodeCallback(BaseCallback):
    def __init__(self, env, total_episodes=100, verbose=0):
        super(EpisodeCallback, self).__init__(verbose)
        self.env = env
        self.total_episodes = total_episodes
        self.current_episode = 0

    def _on_step(self):
        if self.env.step_count >= self.env.max_steps:
            self.current_episode += 1
            if self.current_episode >= self.total_episodes:
                return False
        return True

# Training
print("\n=== Starting DQN Training ===")
env = SumoEnv(Sumo_config)

# Environment check
from stable_baselines3.common.env_checker import check_env
try:
    check_env(env)
except Exception as e:
    print(f"Environment check failed: {e}")
    env.close()
    sys.exit(1)

# DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,  # Adjusted for stability
    gamma=0.99,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.5,
    verbose=1,
    learning_starts=1000,
    train_freq=1,
    batch_size=64,
    target_update_interval=5000
)

# Train
TOTAL_EPISODES = 100
callback = EpisodeCallback(env, total_episodes=TOTAL_EPISODES, verbose=1)
model.learn(total_timesteps=TOTAL_EPISODES * 1000, callback=callback, progress_bar=True)

# Save model
model.save("dqn_sumo")

# Close environment
env.close()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("DQN Training: Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.queue_history, marker='o', linestyle='-', label="Average Queue Length")
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("DQN Training: Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.show()