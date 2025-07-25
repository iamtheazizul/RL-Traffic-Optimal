import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import traci
import time

# Establish path to SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
Sumo_config = [
    'sumo',
    '-c', 'simulation_run_rl.sumocfg',
    '--step-length', ''
    '1',
    '--delay', '0.1'
]

# Hyperparameters
TOTAL_EPISODES = 100
STEPS_PER_EPISODE = 1000
MIN_GREEN_STEPS = 450

# Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0 = keep phase, 1 = switch phase
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            shape=(7,),
            dtype=np.float32
        )
        self.tls_id = "Node2"
        self.min_green_steps = MIN_GREEN_STEPS
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        self.total_steps = STEPS_PER_EPISODE
        self.connection_active = False
        self.connection_label = "default"
        self.valid_ids = False
        self.max_retries = 5
        self.retry_delay = 2

    def reset(self, seed=None, options=None, retry_count=0):
        if self.connection_active:
            try:
                traci.close()
                time.sleep(self.retry_delay)
            except traci.exceptions.TraCIException:
                pass
            finally:
                self.connection_active = False
        try:
            self.connection_label = f"sumo_{id(self)}_{retry_count}"
            traci.start(Sumo_config, label=self.connection_label)
            self.connection_active = True
            traci.simulationStep()
            self._validate_ids()
        except traci.exceptions.TraCIException as e:
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self.reset(seed, options, retry_count + 1)
            else:
                raise e
        self.current_simulation_step = 0
        self.last_switch_step = -self.min_green_steps
        state = self._get_state()
        return state, {}

    def _validate_ids(self):
        try:
            traffic_lights = traci.trafficlight.getIDList()
            detectors = traci.lanearea.getIDList()
            expected_detectors = [
                "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
                "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
            ]
            if self.tls_id not in traffic_lights:
                self.valid_ids = False
                return
            if not all(d in detectors for d in expected_detectors):
                self.valid_ids = False
                return
            self.valid_ids = True
        except traci.exceptions.TraCIException:
            self.valid_ids = False

    def step(self, action):
        self.current_simulation_step += 1
        self._apply_action(action)
        try:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            if current_time < 0:
                return self._get_state(), 0.0, True, False, {"error": "Negative simulation time"}
            if vehicle_count == 0 and self.current_simulation_step < self.total_steps:
                return self._get_state(), 0.0, True, False, {"error": "No vehicles"}
        except traci.exceptions.TraCIException as e:
            return self._get_state(), 0.0, True, False, {"error": str(e)}
        
        new_state = self._get_state()
        reward = self._get_reward(new_state) 
        done = self.current_simulation_step >= self.total_steps
        truncated = False
        info = {}
        return new_state, reward, done, truncated, info

    def _get_state(self):
        if not self.valid_ids:
            return np.zeros(7, dtype=np.float32)
        try:
            q_EB_0 = self._get_queue_length("Node1_2_EB_0")
            q_EB_1 = self._get_queue_length("Node1_2_EB_1")
            q_EB_2 = self._get_queue_length("Node1_2_EB_2")
            q_SB_0 = self._get_queue_length("Node2_7_SB_0")
            q_SB_1 = self._get_queue_length("Node2_7_SB_1")
            q_SB_2 = self._get_queue_length("Node2_7_SB_2")
            current_phase = self._get_current_phase(self.tls_id)
            state = np.array([q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase], dtype=np.float32)
            return state
        except traci.exceptions.TraCIException:
            return np.zeros(7, dtype=np.float32)

    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        return -float(total_queue)

    def _apply_action(self, action):
        if not self.valid_ids:
            return
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                try:
                    program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                    num_phases = len(program.phases)
                    if num_phases == 0:
                        return
                    # toggle between phase groups (0 or 3)
                    current_phase = self._get_current_phase(self.tls_id)
                    phase_group = 0 if current_phase in [0, 1, 2] else 1 if current_phase in [3, 4, 5] else -1
                    next_phase_group = 1 - phase_group
                    next_phase = 3 if next_phase_group == 1 else 0
                    traci.trafficlight.setPhase(self.tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                except traci.exceptions.TraCIException:
                    pass

    def _get_queue_length(self, detector_id):
        try:
            return traci.lanearea.getLastStepVehicleNumber(detector_id)
        except traci.exceptions.TraCIException:
            return 0.0

    def _get_current_phase(self, tls_id):
        try:
            return traci.trafficlight.getPhase(tls_id)
        except traci.exceptions.TraCIException:
            return 0

    def close(self):
        if self.connection_active:
            try:
                traci.close()
                self.connection_active = False
            except traci.exceptions.TraCIException:
                pass

def main():
    env = SumoEnv()
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=5e-4,     # slightly lower learning rate
    #     n_steps=STEPS_PER_EPISODE,            # smaller rollout steps to update more frequently
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,             # higher discount factor for longer-term credit
    #     verbose=1
    # )

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=2.5e-4,   # moderate LR
    #     n_steps=1024,           # collect ~one episode per update
    #     batch_size=64,         
    #     n_epochs=10,
    #     gamma=0.99,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     verbose=1
    # )

    model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    n_steps=STEPS_PER_EPISODE,  # 1000
    batch_size=64,
    n_epochs=10,
    gamma=0.95,
    verbose=1
    )

    episode_history = []
    reward_history = []
    queue_history = []

    print("\n=== Starting PPO Training ===")
    for episode in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        cumulative_reward = 0.0
        total_queue = 0.0
        print(f"\n=== Episode {episode + 1}/{TOTAL_EPISODES} ===")
        try:
            for step in range(STEPS_PER_EPISODE):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += reward
                total_queue += sum(obs[:-1])
                if done or truncated or step == STEPS_PER_EPISODE - 1:
                    break
                if step % 100 == 0:
                    print(f"Step {step}/{STEPS_PER_EPISODE}, Action: {action}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
            model.learn(total_timesteps=STEPS_PER_EPISODE, reset_num_timesteps=False)
        except traci.exceptions.TraCIException as e:
            print(f"Episode {episode + 1} failed: {e}")
            env.close()
            sys.exit(1)
        episode_history.append(episode)
        reward_history.append(cumulative_reward)
        queue_history.append(total_queue / env.current_simulation_step if env.current_simulation_step > 0 else 0)
        print(f"Episode {episode + 1} Summary: Cumulative Reward: {cumulative_reward:.2f}, Avg Queue Length: {queue_history[-1]:.2f}")

    env.close()
    print("\nPPO Training completed.")

    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("RL Training (PPO): Cumulative Reward over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("cumulative_reward_ppo.png")

    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, queue_history, marker='o', linestyle='-', label="Average Queue Length")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.title("RL Training (PPO): Average Queue Length over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("queue_length_ppo.png")

if __name__ == "__main__":
    main()