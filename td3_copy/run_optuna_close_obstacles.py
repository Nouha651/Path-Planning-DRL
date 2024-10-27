import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm  # For progress bar
from td3_copy.code.sim.gym_envirnoment_close_obstacles import ContinuousEnv
from td3_copy.code.agent.TD3 import TD3
from td3_copy.code.agent.ReplayBuffer import ReplayBuffer


def objective(trial):
    # Define hyperparameters with Optuna
    max_episodes = trial.suggest_int("max_episodes", 3000, 8000)
    max_timesteps = trial.suggest_int("max_timesteps", 2000, 5000)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    discount = trial.suggest_float("discount", 0.90, 0.99)
    tau = trial.suggest_float("tau", 0.05, 0.15)
    policy_noise = trial.suggest_float("policy_noise", 0.1, 0.3)
    noise_clip = trial.suggest_float("noise_clip", 0.1, 0.3)
    policy_freq = trial.suggest_int("policy_freq", 1, 3)

    # Print the trial parameters
    print("\nStarting a new trial with parameters:")
    print(f"max_episodes: {max_episodes}, max_timesteps: {max_timesteps}, learning_rate: {learning_rate}")
    print(f"discount: {discount}, tau: {tau}, policy_noise: {policy_noise}")
    print(f"noise_clip: {noise_clip}, policy_freq: {policy_freq}\n")

    # Define model paths
    models_dir = "optuna_models"
    os.makedirs(models_dir, exist_ok=True)
    # Define model paths with trial number
    actor_model_path = os.path.join(models_dir, f"actor_model_trial_{trial.number}.pth")
    critic_model_path = os.path.join(models_dir, f"critic_model_trial_{trial.number}.pth")

    # Initialize environment, agent, and replay buffer
    env = ContinuousEnv()
    state_dim = 2
    action_dim = 2
    buffer_size = 1_000_000

    agent = TD3(state_dim, action_dim, learning_rate=learning_rate, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)
    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    # Training loop
    total_reward = 0
    for episode in tqdm(range(max_episodes), desc="Episodes Progress"):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.select_action(state)
            step_size = np.linalg.norm(action)  # Calculate step size

            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward

            if done:
                break

        # Train the agent if the buffer has enough data
        if replay_buffer.size > 1000:
            agent.train(replay_buffer, batch_size=256)

        total_reward += episode_reward
        trial.report(episode_reward, episode)

        # Prune trial if not promising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save the model
    torch.save(agent.actor.state_dict(), actor_model_path)
    torch.save(agent.critic.state_dict(), critic_model_path)

    avg_reward = total_reward / max_episodes
    print(f"Trial finished with avg_reward: {avg_reward:.2f}")

    return avg_reward


# Run the Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Print the best parameters after the study
    print("\nBest trial:")
    print(f"Value: {study.best_trial.value}")
    print(f"Parameters: {study.best_trial.params}")
