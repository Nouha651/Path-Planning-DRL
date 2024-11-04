import os
import time
import torch
import optuna
from tqdm import tqdm  # For progress bar with time
from TD3.code.sim.gym_envirnment_far_obstacles import ContinuousEnv
from TD3.code.agent.TD3 import TD3
from TD3.code.agent.ReplayBuffer import ReplayBuffer

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune with Optuna
    max_episodes = trial.suggest_int("max_episodes", 3000, 6000)
    max_timesteps = trial.suggest_int("max_timesteps", 1000, 3000)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4)
    discount = trial.suggest_float("discount", 0.90, 0.99)
    tau = trial.suggest_float("tau", 0.01, 0.05)
    policy_noise = trial.suggest_float("policy_noise", 0.05, 0.15)
    noise_clip = trial.suggest_float("noise_clip", 0.05, 0.1)
    policy_freq = trial.suggest_int("policy_freq", 2, 4)

    # Log chosen parameters for this trial
    print(f"\nStarting Trial {trial.number} with hyperparameters: "
          f"max_episodes={max_episodes}, max_timesteps={max_timesteps}, "
          f"learning_rate={learning_rate:.5f}, discount={discount:.2f}, tau={tau:.3f}, "
          f"policy_noise={policy_noise:.3f}, noise_clip={noise_clip:.3f}, "
          f"policy_freq={policy_freq}")

    buffer_size = 1_000_000
    models_dir = "optuna_models"
    os.makedirs(models_dir, exist_ok=True)

    # Paths for baseline and new model files
    baseline_actor_model_path = os.path.join(models_dir, "best_actor_model.pth")
    baseline_critic_model_path = os.path.join(models_dir, "best_critic_model.pth")
    new_actor_model_path = os.path.join(models_dir, f"actor_model_trial_{trial.number}.pth")
    new_critic_model_path = os.path.join(models_dir, f"critic_model_trial_{trial.number}.pth")

    # Initialize environment, agent, and replay buffer
    env = ContinuousEnv()
    state_dim = 2
    action_dim = 2

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

    # Load baseline model for fine-tuning if it exists
    if os.path.exists(baseline_actor_model_path) and os.path.exists(baseline_critic_model_path):
        print(f"Loading baseline model for fine-tuning (Trial {trial.number})...")
        agent.actor.load_state_dict(torch.load(baseline_actor_model_path))
        agent.critic.load_state_dict(torch.load(baseline_critic_model_path))
        agent.actor_target.load_state_dict(torch.load(baseline_actor_model_path))
        agent.critic_target.load_state_dict(torch.load(baseline_critic_model_path))
        print("Baseline model loaded successfully!")
    else:
        print("No baseline model found, starting training from scratch...")

    # Fine-tune the agent and track cumulative reward
    total_reward = 0
    start_time = time.time()

    with tqdm(total=max_episodes, desc=f"Trial {trial.number}", unit="episode") as pbar:
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(max_timesteps):
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
            pbar.set_postfix({
                "Episode": f"{episode + 1}/{max_episodes}",
                "Ep_Reward": f"{episode_reward:.2f}",
                "Time": f"{time.time() - start_time:.2f}s"
            })
            pbar.update(1)

    # Save models for each trial
    torch.save(agent.actor.state_dict(), new_actor_model_path)
    torch.save(agent.critic.state_dict(), new_critic_model_path)
    print(f"Saved model for Trial {trial.number}.")

    # Calculate and return the average reward per episode
    avg_reward = total_reward / max_episodes
    print(f"Trial {trial.number} completed with avg_reward: {avg_reward:.2f}")

    # Update the baseline model if the current trial's reward is higher
    global best_avg_reward
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(agent.actor.state_dict(), baseline_actor_model_path)
        torch.save(agent.critic.state_dict(), baseline_critic_model_path)
        print(f"New baseline model saved with avg_reward: {avg_reward:.2f}")

    return avg_reward

# Global variable to keep track of the best average reward
best_avg_reward = float('-inf')

# Run the Optuna optimization study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # Print the best parameters found by Optuna
    print("Best hyperparameters:", study.best_params)

    # Display top 3 trials for reference
    best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    for i, trial in enumerate(best_trials):
        print(f"Top {i + 1} Trial - Avg Reward: {trial.value:.2f}, Params: {trial.params}")
