# Step 1: Add modules to provide access to specific libraries and functions
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import traci
import logging
import time

# Step 2: Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 3: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo',  # Switch to 'sumo' for training if GUI issues persist
    '-c', 'simulation_run_rl.sumocfg',
    '--step-length', '1.0',
    '--delay', '10'
]

# Step 5: Define Custom SUMO Environment for PPO
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
        self.min_green_steps = 10  # 10s minimum green time
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        self.total_steps = 1000  # 1000s simulation
        self.connection_active = False
        self.connection_label = "default"
        self.valid_ids = False
        self.max_retries = 5  # Increased retry limit
        self.retry_delay = 2  # Increased delay for GUI

    # def reset(self, seed=None, options=None, retry_count=0):
    #     logger.debug(f"Resetting environment (retry {retry_count})")
    #     if self.connection_active:
    #         try:
    #             traci.close()
    #             self.connection_active = False
    #             logger.debug("Closed existing TraCI connection")
    #         except traci.exceptions.TraCIException as e:
    #             logger.warning(f"Failed to close TraCI connection: {e}")
        
    #     try:
    #         self.connection_label = f"sumo_{id(self)}_{retry_count}"
    #         traci.start(Sumo_config, label=self.connection_label)
    #         self.connection_active = True
    #         logger.debug(f"Started new TraCI connection with label: {self.connection_label}")
    #         traci.simulationStep()
    #         current_time = traci.simulation.getTime()
    #         logger.debug(f"Initial simulation time: {current_time}")
    #         self._validate_ids()
    #         vehicle_count = traci.vehicle.getIDCount()
    #         logger.debug(f"Initial vehicle count: {vehicle_count}")
    #         if vehicle_count == 0:
    #             logger.warning("No vehicles in simulation")
    #     except traci.exceptions.TraCIException as e:
    #         logger.error(f"Failed to start TraCI: {e}")
    #         if "connection could be made" in str(e).lower() and retry_count < self.max_retries:
    #             logger.debug(f"Retrying reset (attempt {retry_count + 1}) after {self.retry_delay}s")
    #             time.sleep(self.retry_delay)
    #             return self.reset(seed, options, retry_count + 1)
    #         elif retry_count < self.max_retries:
    #             logger.debug(f"Retrying reset (attempt {retry_count + 1}) after {self.retry_delay}s")
    #             time.sleep(self.retry_delay)
    #             return self.reset(seed, options, retry_count + 1)
    #         else:
    #             logger.error("Max retries reached, switching to 'sumo' for stability")
    #             Sumo_config[0] = 'sumo'
    #             time.sleep(self.retry_delay)
    #             return self.reset(seed, options, 0)
        
    #     self.current_simulation_step = 0
    #     self.last_switch_step = -self.min_green_steps
    #     state = self._get_state()
    #     logger.debug(f"Initial state: {state}")
    #     return state, {}

    def reset(self, seed=None, options=None):
        logger.debug("Resetting environment")

        # Close existing TraCI connection if active
        if self.connection_active:
            try:
                traci.close()
                logger.debug("Closed existing TraCI connection")
                time.sleep(1)  # Wait to ensure SUMO fully closes
            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to close TraCI connection: {e}")
            finally:
                self.connection_active = False

        try:
            # Start new TraCI connection
            self.connection_label = f"sumo_{id(self)}"
            traci.start(Sumo_config, label=self.connection_label)
            self.connection_active = True
            logger.debug(f"Started new TraCI connection with label: {self.connection_label}")

            traci.simulationStep()
            current_time = traci.simulation.getTime()
            logger.debug(f"Initial simulation time: {current_time}")

            self._validate_ids()

            vehicle_count = traci.vehicle.getIDCount()
            logger.debug(f"Initial vehicle count: {vehicle_count}")
            if vehicle_count == 0:
                logger.warning("No vehicles present in simulation at start")

        except traci.exceptions.TraCIException as e:
            logger.error(f"Failed to start TraCI: {e}")
            # Raise exception and let caller decide how to handle it
            raise e

        self.current_simulation_step = 0
        self.last_switch_step = -self.min_green_steps

        state = self._get_state()
        logger.debug(f"State after reset: {state}")

        return state, {}
    
    def _validate_ids(self):
        try:
            traffic_lights = traci.trafficlight.getIDList()
            detectors = traci.lanearea.getIDList()
            expected_detectors = [
                "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
                "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
            ]
            logger.debug(f"Available traffic lights: {traffic_lights}")
            logger.debug(f"Available detectors: {detectors}")
            if self.tls_id not in traffic_lights:
                logger.error(f"Traffic light ID {self.tls_id} not found")
                self.valid_ids = False
                return
            if not all(d in detectors for d in expected_detectors):
                logger.error(f"Some detector IDs not found: {expected_detectors}")
                self.valid_ids = False
                return
            self.valid_ids = True
        except traci.exceptions.TraCIException as e:
            logger.error(f"Failed to validate IDs: {e}")
            self.valid_ids = False

    def step(self, action):
        self.current_simulation_step += 1
        # logger.debug(f"Step {self.current_simulation_step}, Action: {action}")
        self._apply_action(action)
        try:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            # logger.debug(f"Simulation time: {current_time}")
            vehicle_count = traci.vehicle.getIDCount()
            # logger.debug(f"Vehicle count: {vehicle_count}")
            if current_time < 0:
                logger.error("Negative simulation time detected")
                return self._get_state(), 0.0, True, False, {"error": "Negative simulation time"}
            if vehicle_count == 0 and self.current_simulation_step < self.total_steps:
                logger.warning("No vehicles in simulation, terminating episode")
                return self._get_state(), 0.0, True, False, {"error": "No vehicles"}
        except traci.exceptions.TraCIException as e:
            logger.error(f"Simulation step failed: {e}")
            return self._get_state(), 0.0, True, False, {"error": str(e)}
        
        new_state = self._get_state()
        reward = self._get_reward(new_state)
        done = self.current_simulation_step >= self.total_steps
        truncated = False
        info = {}
        # logger.debug(f"New state: {new_state}, Reward: {reward}, Done: {done}")
        return new_state, reward, done, truncated, info

    def _get_state(self):
        if not self.valid_ids:
            logger.warning("Invalid IDs detected, returning zero state")
            return np.zeros(7, dtype=np.float32)
        try:
            detector_Node1_2_EB_0 = "Node1_2_EB_0"
            detector_Node1_2_EB_1 = "Node1_2_EB_1"
            detector_Node1_2_EB_2 = "Node1_2_EB_2"
            detector_Node2_7_SB_0 = "Node2_7_SB_0"
            detector_Node2_7_SB_1 = "Node2_7_SB_1"
            detector_Node2_7_SB_2 = "Node2_7_SB_2"
            
            q_EB_0 = self._get_queue_length(detector_Node1_2_EB_0)
            q_EB_1 = self._get_queue_length(detector_Node1_2_EB_1)
            q_EB_2 = self._get_queue_length(detector_Node1_2_EB_2)
            q_SB_0 = self._get_queue_length(detector_Node2_7_SB_0)
            q_SB_1 = self._get_queue_length(detector_Node2_7_SB_1)
            q_SB_2 = self._get_queue_length(detector_Node2_7_SB_2)
            current_phase = self._get_current_phase(self.tls_id)
            
            state = np.array([q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase], dtype=np.float32)
            # logger.debug(f"State retrieved: {state}")
            return state
        except traci.exceptions.TraCIException as e:
            logger.error(f"Failed to get state: {e}")
            return np.zeros(7, dtype=np.float32)

    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        # logger.debug(f"Reward calculated: {-total_queue}")
        return -float(total_queue)

    def _apply_action(self, action):
        # logger.debug(f"Applying action: {action}")
        if not self.valid_ids:
            logger.warning("Skipping action due to invalid IDs")
            return
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                try:
                    program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                    num_phases = len(program.phases)
                    if num_phases == 0:
                        logger.error("No phases defined for traffic light")
                        return
                    next_phase = (self._get_current_phase(self.tls_id) + 1) % num_phases
                    traci.trafficlight.setPhase(self.tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                    # logger.debug(f"Switched to phase {next_phase}")
                except traci.exceptions.TraCIException as e:
                    logger.error(f"Failed to apply action: {e}")

    def _get_queue_length(self, detector_id):
        try:
            queue_length = traci.lanearea.getLastStepVehicleNumber(detector_id)
            # logger.debug(f"Queue length for {detector_id}: {queue_length}")
            return queue_length
        except traci.exceptions.TraCIException as e:
            logger.warning(f"Failed to get queue length for {detector_id}: {e}")
            return 0.0

    def _get_current_phase(self, tls_id):
        try:
            phase = traci.trafficlight.getPhase(tls_id)
            # logger.debug(f"Current phase for {tls_id}: {phase}")
            return phase
        except traci.exceptions.TraCIException as e:
            logger.warning(f"Failed to get phase for {tls_id}: {e}")
            return 0

    def close(self):
        if self.connection_active:
            try:
                traci.close()
                self.connection_active = False
                # logger.debug("Closed TraCI connection")
            except traci.exceptions.TraCIException as e:
                logger.warning(f"Failed to close TraCI connection: {e}")

def main():
    env = SumoEnv()
    # Validate environment (disabled to avoid GUI issues)
    # try:
    #     check_env(env, warn=True)
    #     logger.debug("Environment check passed")
    # except Exception as e:
    #     logger.error(f"Environment check failed: {e}")
    #     env.close()
    #     sys.exit(1)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        verbose=1
    )
    
    # Lists to record data for plotting
    step_history = []
    reward_history = []
    queue_history = []
    cumulative_reward = 0.0
    
    print("\n=== Starting PPO Training ===")
    total_timesteps = 10000
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        logger.debug("Training completed successfully")
    except traci.exceptions.TraCIException as e:
        logger.error(f"Training failed: {e}")
        env.close()
        sys.exit(1)
    
    # Evaluation loop
    try:
        obs, _ = env.reset()
        cumulative_reward = 0.0
        step_history = []
        reward_history = []
        queue_history = []
        for step in range(total_timesteps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(obs[:-1]))
            # logger.debug(f"Evaluation step {step}: action={action}, reward={reward}, state={obs}, done={done}, truncated={truncated}, info={info}")
            if done or truncated:
                logger.debug(f"Evaluation ended at step {step}, done: {done}, truncated: {truncated}, info: {info}")
                break
    except traci.exceptions.TraCIException as e:
        logger.error(f"Evaluation failed: {e}")
        env.close()
        sys.exit(1)
    
    env.close()
    
    print("\nPPO Training completed.")
    
    # Visualization of Results
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title("RL Training (PPO): Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title("RL Training (PPO): Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
