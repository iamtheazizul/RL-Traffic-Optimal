import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module
import traci

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo',  # Changed from 'sumo-gui' to 'sumo' for non-GUI simulation
    '-c', 'simulation_run_rl.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 5: Define Variables
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
current_phase = 0

# Reinforcement Learning Hyperparameters
TOTAL_EPISODES = 100    # Total number of episodes
STEPS_PER_EPISODE = 1000  # Number of steps per episode
ALPHA = 0.1            # Learning rate
GAMMA = 0.9            # Discount factor
EPSILON = 0.1          # Exploration rate
ACTIONS = [0, 1]       # Action space (0 = keep phase, 1 = switch phase)
Q_table = {}           # Q-table dictionary
MIN_GREEN_STEPS = 100

# Lists to record data for plotting
episode_history = []
reward_history = []
queue_history = []

# Step 6: Define Functions
def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):
    total_queue = sum(state[:-1])  # Exclude current_phase
    reward = -float(total_queue)
    return reward

def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"
    traffic_light_id = "Node2"
    
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)
    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)
    current_phase = get_current_phase(traffic_light_id)
    
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def apply_action(action, tls_id, current_simulation_step, last_switch_step):
    if action == 0:
        return last_switch_step  # No change, return current last_switch_step
    elif action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            return current_simulation_step  # Update last_switch_step
        return last_switch_step  # No phase change, return unchanged
    return last_switch_step  # Fallback, though action is always 0 or 1

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

# Step 7: Episode-based Training Loop
print("\n=== Starting Episode-based Reinforcement Learning ===")
for episode in range(TOTAL_EPISODES):
    # Reset simulation for new episode
    traci.start(Sumo_config)
    last_switch_step = -MIN_GREEN_STEPS  # Initialize locally for episode
    cumulative_reward = 0.0
    total_queue = 0.0
    
    print(f"\n=== Episode {episode + 1}/{TOTAL_EPISODES} ===")
    for step in range(STEPS_PER_EPISODE):
        current_simulation_step = step
        state = get_state()
        action = get_action_from_policy(state)
        last_switch_step = apply_action(action, "Node2", current_simulation_step, last_switch_step)
        traci.simulationStep()
        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward
        total_queue += sum(new_state[:-1])
        update_Q_table(state, action, reward, new_state)
        
        # Print progress every 100 steps within episode
        if step % 100 == 0:
            print(f"Step {step}/{STEPS_PER_EPISODE}, State: {state}, Action: {action}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
    
    # Record episode statistics
    episode_history.append(episode)
    reward_history.append(cumulative_reward)
    queue_history.append(total_queue / STEPS_PER_EPISODE)  # Average queue length per step
    print(f"Episode {episode + 1} Summary: Cumulative Reward: {cumulative_reward:.2f}, Avg Queue Length: {queue_history[-1]:.2f}")
    
    # Print Q-table for the last state of the episode
    print("Current Q-table (last state):")
    for st, qvals in list(Q_table.items())[-5:]:  # Print last 5 states for brevity
        print(f"  {st} -> {qvals}")
    
    # Close simulation for this episode
    traci.close()

# Print final Q-table info
print("\nTraining completed. Final Q-table size:", len(Q_table))
for st, actions in list(Q_table.items())[-5:]:  # Print last 5 for brevity
    print("State:", st, "-> Q-values:", actions)

# Step 8: Visualization of Results
# Plot Cumulative Reward over Episodes
plt.figure(figsize=(10, 6))
plt.plot(episode_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.show()

# Plot Average Queue Length over Episodes
plt.figure(figsize=(10, 6))
plt.plot(episode_history, queue_history, marker='o', linestyle='-', label="Average Queue Length")
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("RL Training: Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.show()