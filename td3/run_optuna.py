import optuna
import torch
import numpy as np
from td3.code.agent.TD3 import TD3
from td3.code.agent.ReplayBuffer import ReplayBuffer
from td3.code.sim.gym_environment import ContinuousEnv
from tqdm import tqdm  # Import tqdm for progress bar

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    max_episodes = trial.suggest_int('max_episodes', 1500, 10000)  # Adjusted range for episodes
    max_timesteps = trial.suggest_int('max_timesteps', 4000, 10000)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    discount = trial.suggest_float("discount", 0.9, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.05)
    policy_noise = trial.suggest_float("policy_noise", 0.1, 0.3)
    noise_clip = trial.suggest_float("noise_clip", 0.3, 0.6)
    policy_freq = trial.suggest_int("policy_freq", 1, 2)
#max action need to be added
    print(
        f"Running trial with parameters: max_episodes={max_episodes}, max_timesteps={max_timesteps}, "
        f"learning_rate={learning_rate}, discount={discount}, tau={tau}, policy_noise={policy_noise}, "
        f"noise_clip={noise_clip}, policy_freq={policy_freq}")

    # Environment setup with max_step_size
    env = ContinuousEnv()  # Pass max_step_size to environment

    # Set state and action dimensions manually
    state_dim = 2  # Assuming state is represented by the agent's x and y positions
    action_dim = 2  # Assuming two continuous actions for x and y movement
    max_action = 1.0  # Normalized continuous action range between -1 and 1

    # Initialize TD3 agent and replay buffer
    agent = TD3(state_dim, action_dim, max_action, discount, tau, policy_noise, noise_clip, policy_freq, learning_rate)
    replay_buffer = ReplayBuffer(1_000_000, state_dim, action_dim)

    episode_rewards = []

    # Progress bar for episodes
    with tqdm(total=max_episodes, desc="Trial Progress", unit="episode") as pbar:
        # Training loop
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(max_timesteps):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                replay_buffer.add(state, action, next_state, reward, done)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Train agent
            if replay_buffer.size > 1000:
                agent.train(replay_buffer)

            episode_rewards.append(episode_reward)

            # Update progress bar with the current episode number
            pbar.set_postfix_str(f"Episode Reward: {episode_reward:.2f}")
            pbar.update(1)

            # Check for trial early stopping
            trial.report(np.mean(episode_rewards[-10:]), episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return np.mean(episode_rewards)


# Optuna study setup
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Output best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
