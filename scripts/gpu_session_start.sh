#!/bin/bash

# gpu_session_start.sh
# Description: Requests an interactive session on a compute node with a specific GPU.
# This script is a convenient way to get a shell where you can run commands
# that require the MI210 GPU, such as running python asdf.py interactively.

# --- SLURM Configuration ---
# Requests the 'gpu' partition
PARTITION="gpu"

# Requests 1 MI210 GPU
GRES="gpu:MI210:4"

# Sets the maximum job time to 10 minutes (adjust as needed, e.g., 4:00:00 for 4 hours)
TIME="10:00:00" 

# Sets the shell to open interactively
SHELL_TYPE="/bin/bash"

# --- Execution ---
echo "Attempting to request an interactive session on a GPU node..."
echo "Resources requested: Partition=${PARTITION}, GPU=${GRES}, Time=${TIME}"

# Execute the srun command to establish the interactive session
srun --partition="$PARTITION" --gres="$GRES" --time="$TIME" --pty "$SHELL_TYPE"

# Note: Once the connection is established, you will be on the compute node.
# Don't forget to run 'source /path/to/your/venv/bin/activate' before running your scripts.
# Type 'exit' to end the session and release the node.
