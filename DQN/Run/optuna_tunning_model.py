import os
import time
import torch
import optuna
from tqdm import tqdm
from environment import PathPlanningEnv
from dqn import DQNAgent

# Objective function for Optuna
def objective(trial):
    # Define hyperparameters for tuning with Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.7, 0.99)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.99999)
    min_epsilon = trial.suggest_float('min_epsilon', 0.01, 0.1)
    step_penalty = trial.suggest_float('step_penalty', -0.2, -0.01)
    obstacle_penalty = trial.suggest_float('obstacle_penalty', -3, -0.5)
    goal_reward = trial.suggest_float('goal_reward', 500, 1000)
    episodes = trial.suggest_int('episodes', 1000, 1500)
    max_steps_per_episode = trial.suggest_int('max_steps_per_episode', 1000, 5000)

    # Display selected parameters for each trial
    print(f"\nStarting Trial {trial.number} with hyperparameters:")
    print(f"Learning rate: {learning_rate}, Gamma: {gamma}, Epsilon decay: {epsilon_decay}, Min epsilon: {min_epsilon}")
    print(f"Step penalty: {step_penalty}, Obstacle penalty: {obstacle_penalty}, Goal reward: {goal_reward}")
    print(f"Episodes: {episodes}, Max steps per episode: {max_steps_per_episode}")

    # Initialize environment and agent with the trial's parameters
    env = PathPlanningEnv(
        grid_size=15,
        start=(0, 0),
        goal=(14, 14),
        step_penalty=step_penalty,
        obstacle_penalty=obstacle_penalty,
        goal_reward=goal_reward,
        random_obstacles=True,
        seed=int(time.time())
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(
        state_dim,
        action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        device=device
    )

    # Model path settings
    model_file = "dqn_model_random.pth"
    new_model_file = f"dqn_model_trial_{trial.number}.pth"
    model_loaded = False

    # Load existing model if available
    if os.path.exists(model_file):
        print(f"Loading existing model for fine-tuning from {model_file}...")
        agent.load_model(model_file)
        model_loaded = True
    else:
        print("No saved model found, starting training from scratch...")

    # Train the model
    total_reward = 0
    start_time = time.time()

    with tqdm(total=episodes, desc=f"Trial {trial.number}", unit="episode") as pbar:
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                steps += 1
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.replay(episode)
                state = next_state
                episode_reward += reward

            # Track cumulative and episode reward
            total_reward += episode_reward

            # Update progress bar with episode details
            pbar.set_postfix({
                "Ep_Reward": f"{episode_reward:.2f}",
                "Steps": f"{steps}",
                "Time": f"{time.time() - start_time:.2f}s"
            })
            pbar.update(1)

    # Save the model for the current trial
    agent.save_model(new_model_file)
    print(f"Model saved for Trial {trial.number}.")

    # Calculate and return average reward per episode
    avg_reward = total_reward / episodes
    print(f"Trial {trial.number} completed with avg_reward: {avg_reward:.2f}")
    return avg_reward

# Run Optuna optimization study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Print the best parameters found by Optuna
    print("Best hyperparameters:", study.best_params)

    # Display top 3 trials for reference
    best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    for i, trial in enumerate(best_trials):
        print(f"Top {i + 1} Trial - Avg Reward: {trial.value:.2f}, Params: {trial.params}")
