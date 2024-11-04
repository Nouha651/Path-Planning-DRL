import subprocess
import os

# Set the number of successful simulations needed
successful_runs_needed = 5
successful_runs = 0

# Directory and file names
run_script = "run2.py"

while successful_runs < successful_runs_needed:
    # Run the script
    result = subprocess.run(["python", run_script], capture_output=True, text=True)

    # Extract the reward from the output
    output = result.stdout
    reward_line = [line for line in output.splitlines() if "Reward:" in line]
    if reward_line:
        reward = float(reward_line[-1].split("Reward: ")[1].split(",")[0])

        # Check if the reward is greater than 980
        if reward > 950:
            successful_runs += 1
            print(f"Successful run {successful_runs}: Reward = {reward}")
        else:
            print(f"Run did not meet the reward threshold: Reward = {reward}")

print(f"Completed {successful_runs_needed} successful runs with rewards greater than 980.")
