from td3.code.sim.gym_environment import ContinuousEnv
from td3.code.agent.TD3 import TD3
from td3.code.agent.ReplayBuffer import ReplayBuffer
from td3.code.viz.viz_screen import Visualize
from td3.constants import *

def main():
    # Set the parameters manually (from your Optuna trial)
    max_episodes = 5826
    max_timesteps = 20000
    learning_rate = 0.00013902671961459994
    discount = 0.9035659227789998
    tau = 0.030098730910611486
    policy_noise = 0.1396707841673583
    noise_clip = 0.46254978250587453
    policy_freq = 1
    buffer_size = 1_000_000  # Set a large replay buffer size
    max_action = 1.0  # Action range in continuous space

    # Initialize environment, agent, and replay buffer
    env = ContinuousEnv()  # Use continuous environment as intended
    state_dim = 2  # x and y positions for continuous movement
    action_dim = 2  # Actions can be continuous in both x and y directions


    # Initialize the TD3 agent with manually set parameters
    agent = TD3(state_dim, action_dim, max_action, learning_rate=learning_rate, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

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

        # Optional: Render the environment (you can disable this during training if not needed)
        env.render()

if __name__ == "__main__":
    main()
