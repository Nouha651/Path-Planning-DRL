import os
import torch
import matplotlib.pyplot as plt
from TD3.code.sim.gym_envirnoment_close_obstacles import ContinuousEnv  # Updated to use close obstacles environment
from TD3.code.agent.TD3 import TD3
from TD3.code.agent.ReplayBuffer import ReplayBuffer


def main():
    # Parameters from trial 6
    max_episodes = 5510
    max_timesteps = 3986
    learning_rate = 0.00044153471293928323
    discount = 0.9412335652245275
    tau = 0.1388318445018274
    policy_noise = 0.22980680659672534
    noise_clip = 0.14713939483858282
    policy_freq = 2

    buffer_size = 1_000_000  # Replay buffer size remains the same

    # Create the 'models' directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Generate filenames based on episodes for saved model
    actor_model_path = os.path.join(models_dir, f"actor_model_close_obstacles_{max_episodes}_episodes.pth")
    critic_model_path = os.path.join(models_dir, f"critic_model_close_obstacles_{max_episodes}_episodes.pth")

    # Initialize environment, agent, and replay buffer
    env = ContinuousEnv()  # Using environment with close obstacles
    state_dim = 2  # x and y positions for continuous movement
    action_dim = 2  # Actions can be continuous in both x and y directions

    # Initialize the TD3 agent with specified parameters
    agent = TD3(state_dim, action_dim, learning_rate=learning_rate, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    # Start training
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

        # Train the agent after each episode if buffer has enough data
        if replay_buffer.size > 1000:
            agent.train(replay_buffer, batch_size=256)

        print(f"Episode {episode + 1}, Total Timesteps: {t + 1}, Reward: {episode_reward:.2f}")

    # Save the trained model
    print("Saving trained model...")
    torch.save(agent.actor.state_dict(), actor_model_path)
    torch.save(agent.critic.state_dict(), critic_model_path)
    print(f"Model saved successfully as {os.path.basename(actor_model_path)} and {os.path.basename(critic_model_path)}")

    # Run a simulation to visualize results after training
    state = env.reset()
    done = False
    path = []  # Store the agent's path

    # Event handler for closing the window
    window_closed = False
    def on_key(event):
        nonlocal window_closed
        if event.key == 'q':
            window_closed = True
            plt.close()

    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    while not done and not window_closed:
        path.append(state)
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        env.render(path=path)

    if not window_closed:
        print("Simulation complete. Close the window to exit.")
        plt.show()

    if window_closed:
        print("Window closed. Terminating the program.")
        exit(0)


if __name__ == "__main__":
    main()
