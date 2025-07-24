import os
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import SUMO-RL environment
from sumo_rl import SumoEnvironment  # Or NetEnv for single TLS env

# Define your SUMO config — point to your net and route files
net_file = "your_net_file.net.xml"            # Replace with your network file path
route_file = "your_route_file.rou.xml"        # Replace with your route file path

# Optional: output file for simulation summary
out_csv = "sumo_rl_results.csv"

# Hyperparameters for training
TOTAL_EPISODES = 100
STEPS_PER_EPISODE = 1000

def main():
    env = SumoEnvironment(
        net_file='network/network_rl.net.xml',
        route_file='simulation_run_rl.rou.xml',
        delta_time=5,
        yellow_time=3,
        min_green=42,
        fixed_ts=False,
        use_gui=False,
        single_agent=True  # ADD THIS
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        n_steps=STEPS_PER_EPISODE,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        verbose=1,
    )
    
    episode_history = []
    reward_history = []
    queue_history = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    print("\n=== Starting SUMO-RL PPO Training ===")
    
    for episode in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        cumulative_reward = 0.0
        total_queue = 0.0
        print(f"\n=== Episode {episode + 1}/{TOTAL_EPISODES} ===")
        
        for step in range(STEPS_PER_EPISODE):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            cumulative_reward += sum(reward.values()) if isinstance(reward, dict) else reward
            
            if done:
                break
            
            if step % 100 == 0:
                print(f"Step {step}/{STEPS_PER_EPISODE}, Cumulative Reward: {cumulative_reward:.2f}")        
        # Learn after each episode rollout
        model.learn(total_timesteps=STEPS_PER_EPISODE, reset_num_timesteps=False)
        
        episode_history.append(episode)
        reward_history.append(cumulative_reward)
        queue_history.append(0)  # Replace with queue metric if you extract it from env
        
        print(f"Episode {episode + 1} Summary: Cumulative Reward: {cumulative_reward:.2f}")
    
    env.close()
    print("PPO Training completed.")
    
    # Save plots
    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("SUMO-RL PPO: Cumulative Reward over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "cumulative_reward_sumo_rl_ppo.png"))
    
    plt.show()


if __name__ == "__main__":
    main()