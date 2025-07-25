# Step 1: Add modules
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Establish path to SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module
import traci

# Step 4: Define SUMO configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'simulation_run_rl.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 5: Open connection
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# State variables
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
q_Node4_5_EB_0 = 0
q_Node4_5_EB_1 = 0
q_Node4_5_EB_2 = 0
current_phase_node2 = 0
current_phase_node5 = 0

# RL Hyperparameters
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1, 2, 3]  # 0: Keep both, 1: Switch Node2, 2: Switch Node5, 3: Switch both

# Q-table
Q_table = {}

# Stability parameters
MIN_GREEN_STEPS = 100
MIN_GREEN_STEPS_NODE5_EB = 150  # Longer min green for Node4_5_EB phase
last_switch_step_node2 = -MIN_GREEN_STEPS
last_switch_step_node5 = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions
# -------------------------

def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):
    """
    Reward function: Higher penalty for Node4_5_EB queues to prioritize left turns at Node5.
    """
    # State: (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, q_Node4_5_EB_0, q_Node4_5_EB_1, q_Node4_5_EB_2, phase_node2, phase_node5)
    q_node2 = sum(state[0:6])  # Node2 queues
    q_node5_eb = sum(state[6:9])  # Node4_5_EB queues
    EB_WEIGHT = 2.0  # Higher weight for Node4_5_EB queues
    reward = -(q_node2 + EB_WEIGHT * q_node5_eb)
    return reward

def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, q_Node4_5_EB_0, q_Node4_5_EB_1, q_Node4_5_EB_2, current_phase_node2, current_phase_node5
    
    # Detector IDs for Node2
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"
    
    # Detector IDs for Node5
    detector_Node4_5_EB_0 = "Node4_5_EB_0"
    detector_Node4_5_EB_1 = "Node4_5_EB_1"
    detector_Node4_5_EB_2 = "Node4_5_EB_2"
    
    # Get queue lengths
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)
    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)
    q_Node4_5_EB_0 = get_queue_length(detector_Node4_5_EB_0)
    q_Node4_5_EB_1 = get_queue_length(detector_Node4_5_EB_1)
    q_Node4_5_EB_2 = get_queue_length(detector_Node4_5_EB_2)
    
    # Get current phases
    current_phase_node2 = get_current_phase("Node2")
    current_phase_node5 = get_current_phase("Node5")
    
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, 
            q_Node4_5_EB_0, q_Node4_5_EB_1, q_Node4_5_EB_2, 
            current_phase_node2, current_phase_node5)

def apply_action(action, tls_id_node2="Node2", tls_id_node5="Node5"):
    """
    Executes actions with bias toward Node4_5_EB at Node5.
    """
    global last_switch_step_node2, last_switch_step_node5, current_simulation_step
    
    def switch_phase(tls_id, last_switch_step, is_node5_eb_phase=False):
        min_green = MIN_GREEN_STEPS_NODE5_EB if is_node5_eb_phase else MIN_GREEN_STEPS
        if current_simulation_step - last_switch_step >= min_green:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            current_phase = get_current_phase(tls_id)
            next_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            return current_simulation_step
        return last_switch_step
    
    # Assume Node5's phase 0 allows Node4_5_EB left turns (adjust if different)
    is_node5_eb_phase = (get_current_phase(tls_id_node5) == 0)
    
    if action == 0:
        # Keep both phases
        return
    elif action == 1:
        # Switch Node2 only
        last_switch_step_node2 = switch_phase(tls_id_node2, last_switch_step_node2)
    elif action == 2:
        # Switch Node5, with longer green for Node4_5_EB phase
        last_switch_step_node5 = switch_phase(tls_id_node5, last_switch_step_node5, is_node5_eb_phase)
    elif action == 3:
        # Switch both
        last_switch_step_node2 = switch_phase(tls_id_node2, last_switch_step_node2)
        last_switch_step_node5 = switch_phase(tls_id_node5, last_switch_step_node5, is_node5_eb_phase)

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

# -------------------------
# Step 8: Learning Loop
# -------------------------

step_history = []
reward_history = []
queue_history = []
queue_node4_5_eb_history = []  # Track Node4_5_EB queues separately

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    traci.simulationStep()
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    update_Q_table(state, action, reward, new_state)
    
    updated_q_vals = Q_table[state]
    
    if step % 1 == 0:
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values: {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-2]))
        queue_node4_5_eb_history.append(sum(new_state[6:9]))  # Node4_5_EB queues
    
    if step % 1000 == 0:
        print("Current Q-table:")
        for st, qvals in Q_table.items():
            print(f"  {st} -> {qvals}")

# -------------------------
# Step 9: Close connection
# -------------------------
traci.close()

# Print final Q-table
print("\nOnline Training completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)

# -------------------------
# Visualization
# -------------------------

# Plot Cumulative Reward
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Node4_5_EB Queue Length
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_node4_5_eb_history, marker='o', linestyle='-', label="Node4_5_EB Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Node4_5_EB Queue Length")
plt.title("RL Training: Node4_5_EB Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()