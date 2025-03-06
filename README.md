# Racetrack Reinforcement Learning  

This project implements reinforcement learning to solve the Racetrack problem using the Monte Carlo method. The agent learns to navigate a simulated racetrack by optimizing its policy based on cumulative rewards. A simple and complex track are used for comparison.  

## Reference to Exercise  
Based on **Exercise 5.12 from Chapter 5** of *Reinforcement Learning: An Introduction (2nd Edition)* by Sutton and Barto, this implementation applies Monte Carlo control to train an agent in episodic exploration and policy improvement.  

## Project Structure  
- `main.py`: Runs training and evaluation of the Monte Carlo agent.  
- `results/main/`: Stores training results for the simple racetrack.  
- `results/modified/`: Stores training results for the complex racetrack.  

## Environment  
The `RacetrackEnv` class defines the environment:  
- **Start positions (`S`)**: Where the car begins.  
- **Track area (`1`)**: Drivable surface.  
- **Finish line (`F`)**: The goal.  
- **Walls (`0`)**: Crashing resets the car to a start position with zero velocity.  
- **Velocity updates**: Acceleration in x and y is limited between 0 and 4.  

## Monte Carlo Learning Agent  
The `MonteCarloAgent` follows an **epsilon-greedy** policy:  
- Collects episodes and updates Q-values using **first-visit Monte Carlo**.  
- Dynamically adjusts policy to improve navigation.  
- Limits episode length to 50 steps to prevent excessive wandering.  

## Training & Evaluation  
- **Training:** 25,000 episodes per track.  
- **Evaluation:** A test run using the learned policy.  

## Running the Experiment  
1. Install dependencies:  
   ```bash
   pip install numpy matplotlib
   ```  
2. Run the script:  
   ```bash
   python main.py
   ```  
3. Results are saved in `results/main` and `results/modified`.  

## Outputs  
- **Track visualizations**: Images of both tracks.  
- **Training progress**: Plots of success rate and rewards over episodes.  
- **Final performance comparison**: Summary statistics of both tracks.  

## Conclusion  
This experiment analyzes how track complexity affects learning speed and success rates. Monte Carlo control provides an effective way to train an agent in a model-free setting without requiring transition probabilities.
