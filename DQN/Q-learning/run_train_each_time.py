import gym
import numpy as np
import torch
from environment import PathPlanningEnv
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon_decay, min_epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((state_dim, action_dim))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        td_target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# Best hyperparameters
best_hyperparams = {
    'learning_rate': 0.23191435641508074,
    'gamma': 0.9342111602858847,
    'epsilon_decay': 0.9165616526268954,
    'min_epsilon': 0.01576629874074416
}

# Initialize the environment with random obstacles
env = PathPlanningEnv(
    grid_size=15,
    start=(0, 0),
    goal=(14, 14),
    step_penalty=-0.1,
    obstacle_penalty=-2,
    goal_reward=1000,
    random_obstacles=True,
    seed=np.random.randint(10, 50)  # Optional: set a seed for reproducibility
)

# State and action dimensions based on grid size
state_dim = env.grid_size ** 2  # Total possible states in the grid
action_dim = env.action_space.n

agent = QLearningAgent(state_dim, action_dim, **best_hyperparams)

# Define the number of episodes and maximum steps per episode
episodes = 1000
max_steps_per_episode = 1000

# Initialize path_history to store the paths for each episode
path_history = []

# Train the model
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    path = []  # Initialize the path for this episode

    while not done and steps < max_steps_per_episode:
        steps += 1
        action = agent.act(state[0] * env.grid_size + state[1])  # Convert 2D state to 1D index
        next_state, reward, done, _ = env.step(action)
        agent.update(state[0] * env.grid_size + state[1], action, reward, next_state[0] * env.grid_size + next_state[1], done)
        state = next_state
        total_reward += reward

        path.append(state)  # Add the current state to the path

    path_history.append(path)  # Append the entire path for this episode
    print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

# Optional: Evaluate the model, fine-tune further, or simply run a few episodes to see its performance
state = env.reset()
done = False
path = []
total_reward = 0
steps = 0

while not done and steps < max_steps_per_episode:
    steps += 1
    action = agent.act(state[0] * env.grid_size + state[1])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    path.append(state)
    total_reward += reward

# Print out the results of the evaluation
print(f"Evaluation completed: Reward: {total_reward}, Steps: {steps}")

# Render the final path on the grid using blue points
env.render(path=path)

# Plot the learning progress if training was done
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
