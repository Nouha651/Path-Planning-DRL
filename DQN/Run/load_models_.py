import torch
import gym
import time
import matplotlib.pyplot as plt
from environment import PathPlanningEnv
from dqn import DQNAgent

# Define the models directory and list all models (assuming each model is saved as "dqn_model_{index}.pth")
#num_models = 29
models_directory = "models"  # Update with the actual path where models are saved

#seed=int(time.time())

# Initialize the environment
env = PathPlanningEnv(grid_size=15, start=(0, 0), goal=(14, 14),
                      step_penalty=-0.1, obstacle_penalty=-2, goal_reward=1000, random_obstacles=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loop through each model
for i in range(0, 20):
    model_file = f"{models_directory}/dqn_model_trial_21.pth"
    print(f"Loading model: {model_file}")

    # Initialize agent and load model
    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.load_model(model_file)

    # Run a simulation with the loaded model
    state = env.reset()
    done = False
    path = []
    total_reward = 0
    steps = 0

    print(f"Running simulation for model {i}... (Press 'q' to quit this simulation)")

    while not done and steps < 4468:  # Using max_steps_per_episode from your best hyperparameters
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1

    # Render and display the path
    env.render(path=path)

    # Check for 'q' key to move to the next model
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)
    plt.show(block=True)

    print(f"Completed simulation for model {i}, Total Reward: {total_reward}, Steps: {steps}\n")

print("All models have been evaluated.")
