import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from TD3.code.sim.gym_envirnoment_close_obstacles import ContinuousEnv
from TD3.code.agent.TD3 import TD3
from TD3.code.agent.RRT_STAR_SMART import RRTStarSmart  # Import RRT* Smart
from torch.utils.tensorboard import SummaryWriter  # Import for TensorBoard

# Define metric calculation functions
def calculate_path_length(path):
    return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))

def calculate_collisions(path, env):
    return sum(1 for point in path if
               any(np.linalg.norm(np.array(point) - obstacle) < env.obstacle_radius for obstacle in env.obstacles))

def calculate_clearance(path, env):
    return sum(min(np.linalg.norm(np.array(point) - obstacle) for obstacle in env.obstacles) for point in path) / len(path)

def main():
    # Set up TensorBoard logging directories
    log_dir = "runs/static_env_comparison"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Define model paths
    models_dir = "optuna_models"
    actor_model_path = os.path.join(models_dir, "old_actor_model_1.pth")
    critic_model_path = os.path.join(models_dir, "old_critic_model_1.pth")

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
        agent.actor.load_state_dict(torch.load(actor_model_path, map_location=torch.device('cpu'), weights_only=True))
        agent.critic.load_state_dict(torch.load(critic_model_path, map_location=torch.device('cpu'), weights_only=True))
        agent.actor_target.load_state_dict(
            torch.load(actor_model_path, map_location=torch.device('cpu'), weights_only=True))
        agent.critic_target.load_state_dict(
            torch.load(critic_model_path, map_location=torch.device('cpu'), weights_only=True))
        print("Model loaded successfully!")
    else:
        print("No saved model found. Please ensure the model files exist.")
        return  # Exit if model files are not available

    # Generate the optimal path using RRT* Smart
    rrt_star = RRTStarSmart(env, max_iterations=2000, search_radius=0.5, epsilon=1.0, rewiring_radius=1.5)
    optimal_path = rrt_star.find_path()
    print("Optimal Path (RRT* Smart):", optimal_path)

    # Calculate RRT* metrics
    rrt_path_length = calculate_path_length(optimal_path)
    rrt_collisions = calculate_collisions(optimal_path, env)
    rrt_clearance = calculate_clearance(optimal_path, env)
    rrt_steps = len(optimal_path)  # Number of steps taken by RRT*

    # Log RRT* metrics to TensorBoard and print to console
    writer.add_scalar("RRT*/Path Length", rrt_path_length, 0)
    writer.add_scalar("RRT*/Collisions", rrt_collisions, 0)
    writer.add_scalar("RRT*/Clearance", rrt_clearance, 0)
    writer.add_scalar("RRT*/Steps", rrt_steps, 0)


    # Run the TD3 simulation
    state = env.reset()
    done = False
    td3_path = []  # Store TD3 path as (x, y) coordinates
    cumulative_reward = 0  # Initialize cumulative reward
    td3_steps = 0  # Initialize step counter for TD3

    # Event handler for closing the window
    window_closed = False

    def on_key(event):
        nonlocal window_closed
        if event.key == 'q':
            window_closed = True
            plt.close()

    env.fig.canvas.mpl_connect('key_press_event', on_key)

    while not done and not window_closed:
        # Store the current position in TD3 path
        td3_path.append((state[0], state[1]))

        # Select action and update state
        action = agent.select_action(state)  # No noise during simulation
        next_state, reward, done = env.step(action)
        state = next_state
        cumulative_reward += reward  # Accumulate reward
        td3_steps += 1  # Increment step counter

        # Render environment with both paths (optimal from RRT* Smart and TD3's path)
        env.render(path=td3_path, optimal_path=optimal_path)

    # Add the last position to the TD3 path if done or window was closed
    if done or window_closed:
        td3_path.append((state[0], state[1]))
        env.render(path=td3_path, optimal_path=optimal_path)

    # Calculate TD3 metrics
    td3_path_length = calculate_path_length(td3_path)
    td3_collisions = calculate_collisions(td3_path, env)
    td3_clearance = calculate_clearance(td3_path, env)

    # Log TD3 metrics to TensorBoard and print to console
    writer.add_scalar("TD3/Path Length", td3_path_length, 0)
    writer.add_scalar("TD3/Collisions", td3_collisions, 0)
    writer.add_scalar("TD3/Clearance", td3_clearance, 0)
    writer.add_scalar("TD3/Steps", td3_steps, 0)
    writer.add_scalar("TD3/Cumulative Reward", cumulative_reward, 0)
    # Output the metrics in a comparison table format
    print("\nComparison of Metrics:")
    print("=" * 40)
    print(f"{'Metric':<20} | {'RRT* Smart':<10} | {'TD3':<10}")
    print("=" * 40)
    print(f"{'Path Length':<20} | {rrt_path_length:<10.4f} | {td3_path_length:<10.4f}")
    print(f"{'Collisions':<20} | {rrt_collisions:<10} | {td3_collisions:<10}")
    print(f"{'Clearance':<20} | {rrt_clearance:<10.4f} | {td3_clearance:<10.4f}")
    print(f"{'Steps Taken':<20} | {rrt_steps:<10} | {td3_steps:<10}")
    print(f"{'Cumulative Reward':<20} | {'N/A':<10} | {cumulative_reward:<10.4f}")
    print("=" * 40)

    # Visualize metrics with bar charts in TensorBoard
    metrics = {
        "Path Length": [rrt_path_length, td3_path_length],
        "Collisions": [rrt_collisions, td3_collisions],
        "Clearance": [rrt_clearance, td3_clearance],
        "Steps": [rrt_steps, td3_steps]  # Number of steps taken by each model
    }

    for metric_name, values in metrics.items():
        fig, ax = plt.subplots(figsize=(4, 3))  # Adjust figure size for bar charts
        ax.bar(["RRT* Smart", "TD3"], values, color=["#1f77b4", "#ff7f0e"], width=0.3)  # Set width for bar thickness
        ax.set_title(f"{metric_name} Comparison")
        ax.set_ylabel(metric_name)
        writer.add_figure(f"{metric_name}_Comparison", fig)
        plt.close(fig)  # Close the figure after logging to save memory

    # Close TensorBoard writer
    writer.close()
    print("Metrics logged to TensorBoard as bar charts and scalar.")

    print("Simulation complete.")
    plt.show(block=True)  # Keeps the window open until closed by user


if __name__ == "__main__":
    main()
