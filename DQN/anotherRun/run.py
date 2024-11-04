import gym
import numpy as np
from environment import PathPlanningEnv
from dqn import DQNAgent
import torch
import os
import matplotlib.pyplot as plt

# Best hyperparameters found
best_hyperparams = {
    'learning_rate': 0.0007420972559745429,
    'gamma': 0.940674682502387,
    'epsilon_decay': 0.9919047450722595,
    'min_epsilon': 0.0461282329100503,
    'step_penalty': -0.32469547348467714,
    'obstacle_penalty': -3.1359550476685087,
    'goal_reward': 996.1440163198091,
    'episodes': 1801,
    'max_steps_per_episode': 4247
}

# Initialize the environment with random obstacles
env = PathPlanningEnv(
    grid_size=15,
    start=(0, 0),
    goal=(14, 14),
    step_penalty=best_hyperparams['step_penalty'],
    obstacle_penalty=best_hyperparams['obstacle_penalty'],
    goal_reward=best_hyperparams['goal_reward'],
    random_obstacles=True,  # Use random obstacles for generalization
    seed=None  # Seed is None for varied obstacle layouts
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

agent = DQNAgent(state_dim, action_dim, learning_rate=best_hyperparams['learning_rate'], gamma=best_hyperparams['gamma'],
                 epsilon=1.0, epsilon_decay=best_hyperparams['epsilon_decay'], min_epsilon=best_hyperparams['min_epsilon'], device=device)

# Define the number of episodes and maximum steps per episode
episodes = best_hyperparams['episodes']
max_steps_per_episode = best_hyperparams['max_steps_per_episode']

# Initialize path_history to avoid NameError
path_history = []

# Check if a saved model exists and load it
model_file = "generalized_dqn_model.pth"
model_loaded = False  # Flag to check if the model was loaded

if os.path.exists(model_file):
    print("Loading existing model...")
    agent.load_model(model_file)
    model_loaded = True
else:
    print("No saved model found, training a new one...")

# Train the model only if no model was loaded
if not model_loaded:
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        path = []  # Initialize the path for this episode

        while not done and steps < max_steps_per_episode:
            steps += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.replay(episode)
            state = next_state
            total_reward += reward

            path.append(state)  # Add the current state to the path

        path_history.append(path)  # Append the entire path for this episode
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

        if episode % 10 == 0:
            agent.update_target_model()

    # Save the trained model
    agent.save_model(model_file)
    print("Model saved.")

# If model was loaded, you can evaluate it or continue fine-tuning
else:
    print("Model loaded. You can now evaluate or continue fine-tuning the model.")
    # Optional: Evaluate the model, fine-tune further, or simply run a few episodes to see its performance
    state = env.reset()
    done = False
    path = []
    total_reward = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        steps += 1
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        path.append(state)
        total_reward += reward

    # Print out the results of the evaluation
    print(f"Evaluation completed: Reward: {total_reward}, Steps: {steps}")

    # Render the final path on the grid using blue points
    env.render(path=path)

# Plot the learning progress if training was done
if not model_loaded:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([episode for episode in range(episodes)], [total_reward for _ in range(episodes)])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot([episode for episode in range(episodes)], [agent.epsilon for _ in range(episodes)])
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.show()

# Plot the path from the last episode
if path_history:
    last_path = path_history[-1]
    env.render(path=last_path)
