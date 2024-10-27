import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from td3_copy.code.sim.gym_envirnoment_close_obstacles import ContinuousEnv
from td3_copy.code.agent.TD3 import TD3

def main():
    # Define model paths
    models_dir = "models"
    actor_model_path = os.path.join(models_dir, "actor_model_with_close_obstacles.pth")
    critic_model_path = os.path.join(models_dir, "critic_model_with_close_obstacles.pth")

    # Initialize environment and agent
    env = ContinuousEnv()  # Use the environment with obstacles
    state_dim = 2  # x and y positions for continuous movement
    action_dim = 2  # Actions can be continuous in both x and y directions

    # Fixed hyperparameters for agent initialization
    learning_rate = 0.00017402548614362618
    discount = 0.9480861259418645
    tau = 0.13427117331677685
    policy_noise = 0.19984110422510556
    noise_clip = 0.16910923828314897
    policy_freq = 1

    # Initialize the TD3 agent
    agent = TD3(state_dim, action_dim, learning_rate=learning_rate, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    # Load the trained model
    if os.path.exists(actor_model_path) and os.path.exists(critic_model_path):
        print("Loading saved model for evaluation...")
        agent.actor.load_state_dict(torch.load(actor_model_path, weights_only=True))
        agent.critic.load_state_dict(torch.load(critic_model_path, weights_only=True))
        agent.actor_target.load_state_dict(torch.load(actor_model_path, weights_only=True))
        agent.critic_target.load_state_dict(torch.load(critic_model_path, weights_only=True))
        print("Model loaded successfully!")
    else:
        print("No saved model found. Please ensure the model files exist.")
        return  # Exit if model files are not available

    # Run the simulation
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
        path.append(state)  # Save the current position to the path
        action = agent.select_action(state)  # No noise during simulation
        next_state, reward, done = env.step(action)
        state = next_state
        env.render(path=path)  # Render the environment with the path

    if not window_closed:
        print("Simulation complete. Close the window to exit.")
        plt.show()

    if window_closed:
        print("Window closed. Terminating the program.")
        exit(0)

if __name__ == "__main__":
    main()
