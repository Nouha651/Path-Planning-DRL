import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from td3_copy.code.sim.gym_envirnment_far_obstacles import ContinuousEnv
from td3_copy.code.agent.TD3 import TD3
from td3_copy.code.agent.ReplayBuffer import ReplayBuffer

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune with Optuna
    max_episodes = trial.suggest_int("max_episodes", 3000, 6000)  # Range for fine-tuning episodes
    max_timesteps = trial.suggest_int("max_timesteps", 1000, 4000)  # Range for fine-tuning timesteps
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    discount = 0.9480861259418645  # Keep discount fixed for now
    tau = trial.suggest_float("tau", 0.05, 0.15)
    policy_noise = trial.suggest_float("policy_noise", 0.05, 0.3)
    noise_clip = trial.suggest_float("noise_clip", 0.1, 0.3)
    policy_freq = trial.suggest_int("policy_freq", 1, 4)

    buffer_size = 1_000_000  # Replay buffer size remains the same
    models_dir = "models"
    actor_model_path = os.path.join(models_dir, f"actor_model_with_far_obstacles.pth")
    critic_model_path = os.path.join(models_dir, f"critic_model_with_far_obstacles.pth")

    # Initialize environment, agent, and replay buffer
    env = ContinuousEnv()  # Environment with obstacles
    state_dim = 2  # x and y positions for continuous movement
    action_dim = 2  # Actions can be continuous in both x and y directions

    # Initialize the TD3 agent with Optuna-suggested parameters
    agent = TD3(
        state_dim,
        action_dim,
        learning_rate=learning_rate,
        discount=discount,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_freq=policy_freq
    )

    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    # Load the trained model if available for fine-tuning
    if os.path.exists(actor_model_path) and os.path.exists(critic_model_path):
        print("Loading saved model for fine-tuning...")
        agent.actor.load_state_dict(torch.load(actor_model_path))
        agent.critic.load_state_dict(torch.load(critic_model_path))
        agent.actor_target.load_state_dict(torch.load(actor_model_path))
        agent.critic_target.load_state_dict(torch.load(critic_model_path))
        print("Model loaded successfully!")
    else:
        print("No saved model found, starting training from scratch...")

    # Fine-tune the agent and track cumulative reward
    total_reward = 0
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            # Select action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # Add experience to replay buffer
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Train the agent if the buffer has enough data
        if replay_buffer.size > 1000:
            agent.train(replay_buffer, batch_size=256)

        total_reward += episode_reward

    # Return the total reward for Optuna to maximize
    return total_reward

# Run the Optuna optimization study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Run 20 trials; you can increase if needed

    # Print the best parameters found by Optuna
    print("Best hyperparameters:", study.best_params)
