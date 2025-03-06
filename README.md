# Racetrack Reinforcement Learning

This project implements a reinforcement learning approach to solving the Racetrack problem using the Monte Carlo method. The agent learns to navigate a simulated racetrack environment by optimizing its policy based on cumulative rewards. The environment includes a simple track and a more complex track for comparison.

## Reference to Exercise
This implementation is based on **Exercise 5.12 from Chapter 5** of the *Reinforcement Learning: An Introduction (2nd Edition)* by Sutton and Barto. The exercise focuses on solving the Racetrack problem using Monte Carlo control methods, where the agent learns optimal driving strategies through episodic exploration and policy improvement.

## Project Structure

- `main.py`: The main script that runs the training and evaluation of the Monte Carlo agent on both a simple and a complex racetrack.
- `results/main/`: Stores training results and visualizations for the simple racetrack.
- `results/modified/`: Stores training results and visualizations for the complex racetrack.

## How It Works

### Environment
The `RacetrackEnv` class defines the racetrack where the agent moves based on acceleration actions. Key features include:
- **Start positions (`S`)**: The locations where the car can start.
- **Track area (`1`)**: Drivable parts of the track.
- **Finish line (`F`)**: The goal to reach.
- **Walls (`0`)**: Areas where the car crashes if entered.
- **Velocity updates**: The agent can accelerate in both x and y directions, with velocity clamped between 0 and 4.
- **Crash handling**: If a car crashes into a wall, it is reset to a random start position with zero velocity.

### Monte Carlo Learning Agent
The `MonteCarloAgent` uses an **epsilon-greedy** approach to balance exploration and exploitation. The agent:
- Collects episodes by running a policy.
- Updates Q-values based on the **first-visit Monte Carlo method**.
- Adjusts policy dynamically to improve navigation towards the finish line.
- Limits episode length to 50 steps to prevent excessive wandering.

### Training & Evaluation
- **Training:** The agent is trained for **25,000 episodes** on each track.
- **Evaluation:** A final test episode is run using the learned policy to measure success.

## Key Features
- **Results Saving**: 
  - Training progress and results are automatically saved in `results/main/` and `results/modified/`.
  - Images and plots are cleared from these directories before new training runs.
- **Comparison Between Simple and Complex Tracks**:
  - The success rate and episode length are tracked to evaluate how complexity affects learning.
  - The simple track has a more straightforward path, while the complex track includes obstacles and turns.

## Running the Experiment
1. Ensure dependencies are installed:
   ```bash
   pip install numpy matplotlib
   ```
2. Run the script:
   ```bash
   python main.py
   ```
3. The results, including training progress and policy performance, will be saved in the respective directories.

## Expected Outputs
- **Track Visualizations:** Images of the simple and complex tracks.
- **Training Progress:** Plots showing success rate and average rewards over episodes.
- **Final Performance Comparison:** A printed summary comparing the performance on both tracks.

## Conclusion
This project explores reinforcement learning in a grid-based racetrack environment. By comparing simple and complex tracks, we analyze how environmental difficulty influences learning speed and success rates. The Monte Carlo method provides an effective way to train an agent in a model-free environment without requiring transition probabilities.

