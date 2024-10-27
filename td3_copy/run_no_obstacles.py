import os
import torch
import matplotlib.pyplot as plt
from td3_copy.code.sim.gym_environment_no_obstacle import ContinuousEnv
from td3_copy.code.agent.TD3 import TD3
from td3_copy.code.agent.ReplayBuffer import ReplayBuffer


def main():
    # Assign the values from trial 25 to the corresponding parameters
    max_episodes = 7708
    max_timesteps = 5630
    learning_rate = 0.00017402548614362618
    discount = 0.9480861259418645
    tau = 0.13427117331677685
    policy_noise = 0.19984110422510556
    noise_clip = 0.16910923828314897
    policy_freq = 1

    buffer_size = 1_000_000  # Replay buffer size remains the same

    # Create the 'models' directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Generate filenames6+ using max_episodes
    actor_model_path = os.path.join(models_dir, f"actor_model_no_obstacles_{max_episodes}_episodes.pth")
    critic_model_path = os.path.join(models_dir, f"critic_model_no_obstacles_{max_episodes}_episodes.pth")

    # Initialize environment, agent, and replay buffer
    env = ContinuousEnv()  # Use continuous environment as intended
    state_dim = 2  # x and y positions for continuous movement
    action_dim = 2  # Actions can be continuous in both x and y directions

    # Initialize the TD3 agent
    agent = TD3(state_dim, action_dim, learning_rate=learning_rate, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    # Check if the model exists and load it if it does
    if os.path.exists(actor_model_path) and os.path.exists(critic_model_path):
        print("Loading saved model...")
        agent.actor.load_state_dict(torch.load(actor_model_path, weights_only=True))
        agent.critic.load_state_dict(torch.load(critic_model_path, weights_only=True))
        agent.actor_target.load_state_dict(torch.load(actor_model_path, weights_only=True))  # Load into target networks as well
        agent.critic_target.load_state_dict(torch.load(critic_model_path, weights_only=True))

        print("Model loaded successfully!")
    else:
        print("No saved model found, starting training...")
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            print(f"Starting Episode {episode + 1}")

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

            # Train the agent after each episode (if the buffer has enough data)
            if replay_buffer.size > 1000:
                agent.train(replay_buffer, batch_size=256)

            print(f"Episode {episode + 1}, Total Timesteps: {t + 1}, Reward: {episode_reward:.2f}")

        # Save the trained model with a name based on the number of episodes
        print("Saving model...")
        torch.save(agent.actor.state_dict(), actor_model_path)
        torch.save(agent.critic.state_dict(), critic_model_path)
        print(f"Model saved successfully under {models_dir} as {os.path.basename(actor_model_path)} and {os.path.basename(critic_model_path)}")

    # After loading the model or training, run the simulation to show the path from the agent to the goal
    state = env.reset()
    done = False
    path = []  # Store the agent's path

    # Add a flag to detect if the window was closed
    window_closed = False

    def on_key(event):
        nonlocal window_closed
        if event.key == 'q':
            window_closed = True
            plt.close()

    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    while not done and not window_closed:
        path.append(state)  # Save the current position to the path
        action = agent.select_action(state)  # No noise during simulation
        next_state, reward, done = env.step(action)
        state = next_state
        env.render(path=path)  # Render the environment with the path

    # Keep the window open after the simulation
    if not window_closed:
        print("Simulation complete. Close the window to exit.")
        plt.show()  # This will keep the window open until manually closed

    # Exit the program if the window was closed
    if window_closed:
        print("Window closed. Terminating the program.")
        exit(0)


if __name__ == "__main__":
    main()
