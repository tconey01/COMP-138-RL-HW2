import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import time

# Ensure directories exist and clear old results
def setup_result_directories():
    for dir_path in ["results/main", "results/modified"]:
        os.makedirs(dir_path, exist_ok=True)
        # Remove existing PNG files to avoid confusion with old results
        for f in glob.glob(f"{dir_path}/*.png"):
            os.remove(f)
    
    # Add a timestamp to this run for unique filenames
    timestamp = int(time.time())
    return timestamp

# ================== RACETRACK ENVIRONMENT ==================

class RacetrackEnv:
    def __init__(self, track_grid):
        self.track = np.array(track_grid)
        self.start_positions = self.get_positions('S')
        self.finish_positions = self.get_positions('F')
        
        # For tracking time-step metrics
        self.reward_history = []
        self.velocity_history = []
        self.position_history = []

    def get_positions(self, symbol):
        return [(r, c) for r in range(self.track.shape[0]) 
                for c in range(self.track.shape[1]) if self.track[r, c] == symbol]

    def reset(self):
        self.position = random.choice(self.start_positions)
        self.velocity = (0, 0)
        
        # Reset history trackers
        self.reward_history = []
        self.velocity_history = [self.velocity]
        self.position_history = [self.position]
        
        return self.position, self.velocity

    def bounds_check(self, row, col):
        if row < 0 or row >= self.track.shape[0] or col < 0 or col >= self.track.shape[1]:
            return False
        return self.track[row, col] != '0'  

    def step(self, action):
        ax, ay = action
        vx, vy = self.velocity
        vx = np.clip(vx + ax, 0, 4)
        vy = np.clip(vy + ay, 0, 4)

        if np.random.rand() < 0.1:
            vx, vy = self.velocity

        row, col = self.position
        new_row, new_col = row - vy, col + vx  

        crashed = False
        if not self.bounds_check(new_row, new_col):
            new_row, new_col = random.choice(self.start_positions)
            vx, vy = (0, 0)
            crashed = True

        self.position = (new_row, new_col)
        self.velocity = (vx, vy)
        
        # Track metrics history
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)

        reward = -1
        if crashed:
            reward = -5
        done = (new_row, new_col) in self.finish_positions
        if done:
            reward = 10
            
        self.reward_history.append(reward)

        return (self.position, self.velocity), reward, done

# ================== MONTE CARLO CONTROL AGENT ==================

class MonteCarloAgent:
    def __init__(self, env, epsilon=0.1, discount_factor=1.0):
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(list)

        # For tracking episode-level performance
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []

    def generate_episode(self):
        state = self.env.reset()
        episode = []
        total_reward = 0
        steps = 0
        success = False

        while True:
            if np.random.rand() < self.epsilon:
                action = random.choice(actions)
            else:
                action = max(self.Q[state], key=self.Q[state].get, default=random.choice(actions))

            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                success = True
                break

            if steps >= 50:  
                break

        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.success_history.append(success)
            
        return episode

    def update_Q(self, episode):
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = self.discount_factor * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])

    def get_policy(self):
        """Extract a deterministic policy from the Q-values."""
        policy = {}
        for state in self.Q:
            if self.Q[state]:  # If we have Q-values for this state
                policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy

    def train(self, num_episodes=25000):
        for i in range(num_episodes):
            episode = self.generate_episode()
            self.update_Q(episode)
            
            if (i+1) % 5000 == 0 or i == num_episodes-1:
                print(f"Training Progress: {i+1}/{num_episodes} episodes")

    def run_test_episode(self, policy=None):
        """Run a test episode using the provided policy or the current learned policy."""
        if policy is None:
            policy = self.get_policy()
            
        state = self.env.reset()
        total_reward = 0
        steps = 0
        success = False
        
        while steps < 50:
            # Use the policy if state is known, otherwise random
            if state in policy:
                action = policy[state]
            else:
                action = random.choice(actions)
                
            state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                success = True
                break
                
        return {
            'reward_history': self.env.reward_history,
            'velocity_history': self.env.velocity_history,
            'position_history': self.env.position_history,
            'success': success,
            'total_reward': total_reward,
            'steps': steps
        }

# ================== SIMPLIFIED VISUALIZATION FUNCTIONS ==================

def visualize_track(track, save_path, title="Track Layout"):
    """Visualize the track layout."""
    track_array = np.array(track)
    rows, cols = track_array.shape
    
    # Create a colormap for the track
    cmap = {
        '0': '#2C3E50',  # Dark blue-gray for walls
        '1': '#ECF0F1',  # Light gray for track
        'S': '#2ECC71',  # Green for start
        'F': '#E74C3C'   # Red for finish
    }
    
    # Create a grid for the visualization
    plt.figure(figsize=(min(12, cols+2), min(10, rows+2)))
    
    # Plot each cell with appropriate color
    for r in range(rows):
        for c in range(cols):
            cell = track_array[r, c]
            color = cmap[cell]
            plt.fill_between([c, c+1], [rows-r, rows-r], [rows-(r+1), rows-(r+1)], 
                             color=color, edgecolor='black', linewidth=1)
            
            # Add text labels for Start and Finish
            if cell in ['S', 'F']:
                plt.text(c+0.5, rows-(r+0.5), cell, ha='center', va='center', 
                         color='white', fontweight='bold', fontsize=12)
    
    # Set plot properties
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    plt.xticks(np.arange(0, cols+1, 1))
    plt.yticks(np.arange(0, rows+1, 1))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap['0'], edgecolor='black', label='Wall'),
        Patch(facecolor=cmap['1'], edgecolor='black', label='Track'),
        Patch(facecolor=cmap['S'], edgecolor='black', label='Start'),
        Patch(facecolor=cmap['F'], edgecolor='black', label='Finish')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_reward_vs_timestep(episode_data, save_path, title_prefix=""):
    """Plot reward vs timestep."""
    plt.figure(figsize=(12, 6))
    
    reward_history = episode_data['reward_history']
    cumulative_reward = np.cumsum(reward_history)
    
    # Plot both rewards on same subplot with two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Instantaneous reward
    line1, = ax1.plot(reward_history, 'b-o', markersize=4, label='Reward per Step')
    ax1.set_ylabel('Reward per Step', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(min(reward_history)-1, max(reward_history)+1)
    
    # Cumulative reward
    line2, = ax2.plot(cumulative_reward, 'r-', linewidth=2, label='Cumulative Reward')
    ax2.set_ylabel('Cumulative Reward', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Create combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.title(f'{title_prefix}Rewards Over Time', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add episode summary info
    success_text = "Success" if episode_data['success'] else "Failure"
    plt.figtext(0.5, 0.01, 
               f"Total Steps: {episode_data['steps']}, "
               f"Total Reward: {episode_data['total_reward']}, Outcome: {success_text}", 
               ha="center", fontsize=12, bbox={"facecolor":"#F9E79F", "alpha":0.7, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_training_progress(success_history, episode_lengths, episode_rewards, save_path, title_prefix=""):
    """Plot the training progress (success rate and rewards) - this is the key plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Success Rate (with running average)
    window_size = min(500, len(success_history)//10)
    x = np.arange(len(success_history))
    rolling_success = np.convolve(success_history, np.ones(window_size)/window_size, mode='valid')
    rolling_x = x[window_size-1:]
    
    axes[0].plot(x, np.cumsum(success_history) / (x + 1), 'b-', alpha=0.3, label="Cumulative Success Rate")
    axes[0].plot(rolling_x, rolling_success, 'b-', linewidth=2, label=f"{window_size}-Episode Rolling Average")
    axes[0].set_title(f"{title_prefix}Success Rate Over Training", fontsize=14)
    axes[0].set_xlabel("Episodes", fontsize=12)
    axes[0].set_ylabel("Success Rate", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.05)
    
    # Episode Rewards
    window_size = min(500, len(episode_rewards)//10)
    rolling_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    
    axes[1].plot(rolling_x, rolling_rewards, 'g-', linewidth=2, label="Episode Reward")
    axes[1].set_title(f"{title_prefix}Average Reward per Episode", fontsize=14)
    axes[1].set_xlabel("Episodes", fontsize=12)
    axes[1].set_ylabel("Reward", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(fontsize=10)
    
    # Add summary statistics
    final_success_rate = sum(success_history[-1000:]) / 1000  # Last 1000 episodes
    avg_episode_reward = np.mean(episode_rewards[-1000:])  # Last 1000 episodes
    
    plt.figtext(0.5, 0.01, 
               f"Final Statistics (Last 1000 Episodes):\n"
               f"Success Rate: {final_success_rate:.2f} | "
               f"Avg. Reward: {avg_episode_reward:.2f}", 
               ha="center", fontsize=12, bbox={"facecolor":"#F9E79F", "alpha":0.7, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_experiment(track, save_path_prefix, track_name, num_episodes=25000):
    """Run a simplified experiment including training and essential visualizations."""
    # Visualize the track
    track_viz_path = f"{save_path_prefix}_track.png"
    visualize_track(track, track_viz_path, f"{track_name} Track Layout")
    print(f"Track visualization saved to {track_viz_path}")
    
    # Run the experiment
    env = RacetrackEnv(track)
    agent = MonteCarloAgent(env)
    agent.train(num_episodes)
    
    # Plot training progress (key plot)
    training_path = f"{save_path_prefix}_training_progress.png"
    plot_training_progress(
        agent.success_history,
        agent.episode_lengths,
        agent.episode_rewards,
        training_path,
        f"{track_name} Track: "
    )
    print(f"Training progress plot saved to {training_path}")
    
    # Run a test episode with the final policy
    test_episode = agent.run_test_episode()
    
    # Plot reward vs timestep
    reward_path = f"{save_path_prefix}_reward_vs_timestep.png"
    plot_reward_vs_timestep(
        test_episode, 
        reward_path,
        f"{track_name} Track: "
    )
    print(f"Reward vs timestep plot saved to {reward_path}")
    
    return agent.episode_rewards, agent.episode_lengths, agent.success_history

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Set up result directories and get timestamp
    timestamp = setup_result_directories()
    
    simple_track = [
        ['0', '0', '0', 'F', '0'],
        ['S', '1', '1', '1', '1'],
        ['S', '1', '0', '0', '1'],
        ['S', '1', '1', '1', '1']
    ]

    complex_track = [
        ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', 'F', 'F', 'F', 'F'],
        ['S', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '1', '1', '1', 'F'],
        ['S', '1', '0', '0', '0', '0', '0', '1', '0', '1', '1', '1', '0', '1', '0', '1', '1'],
        ['S', '1', '1', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1', '0', '0', '1'],
        ['S', '1', '0', '0', '0', '1', '0', '0', '1', '1', '0', '0', '0', '1', '1', '0', '1'],
        ['0', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '0', '0', '1'],
        ['0', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0', '1', '1', '1', '1', '1', '1'],
        ['0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '1'],
        ['0', '0', '0', '0', '0', '1', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1'],
        ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '0', '0', '0', '0', '1', '1', '1'],
        ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
    ]

    actions = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),  (0, 0),  (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    # Set the number of episodes to 25000
    NUM_EPISODES = 25000

    print("\nTraining on Simple Track...")
    simple_rewards, simple_lengths, simple_success = run_experiment(
        simple_track, 
        "results/main/simple_track", 
        "Simple", 
        NUM_EPISODES
    )

    print("\nTraining on Complex Track...")
    complex_rewards, complex_lengths, complex_success = run_experiment(
        complex_track, 
        "results/modified/complex_track", 
        "Complex", 
        NUM_EPISODES
    )

    # Compare the performance of both tracks
    simple_final_success = sum(simple_success[-1000:]) / 1000
    complex_final_success = sum(complex_success[-1000:]) / 1000
    
    print("\nPerformance Comparison:")
    print(f"Simple Track - Success Rate: {simple_final_success:.2f}, Avg Episode Length: {np.mean(simple_lengths[-1000:]):.2f}")
    print(f"Complex Track - Success Rate: {complex_final_success:.2f}, Avg Episode Length: {np.mean(complex_lengths[-1000:]):.2f}")

    print(f"\nTraining complete! Results saved in results/main/ and results/modified/")
    print(f"Number of episodes trained: {NUM_EPISODES}")