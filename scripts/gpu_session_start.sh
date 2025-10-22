#!/bin/bash

# gpu_session_start.sh
# Description: Requests an interactive session on a compute node with a specific GPU.
# This script is a convenient way to get a shell where you can run commands
# that require the MI210 GPU, such as running python asdf.py interactively.

# --- SLURM Configuration ---
PARTITION="gpu"           # Requests the 'gpu' partition
GRES="gpu:MI210:2"        # Requests 1 MI210 GPU
TIME="24:00:00"           # Maximum job time (e.g., 4:00:00 for 4 hours)
SHELL_TYPE="/bin/bash"    # Shell type for the interactive session

# --- MIOpen Configuration ---
CACHE_DIR="$HOME/.miopen_cache"
USER_DB_DIR="$HOME/.miopen_user_db"

echo "🔍 Checking MIOpen directories..."

# Create directories if they don't exist
if [ ! -d "$CACHE_DIR" ]; then
    echo "📁 Creating folder: $CACHE_DIR"
    mkdir -p "$CACHE_DIR"
else
    echo "✅ Folder already exists: $CACHE_DIR"
fi

if [ ! -d "$USER_DB_DIR" ]; then
    echo "📁 Creating folder: $USER_DB_DIR"
    mkdir -p "$USER_DB_DIR"
else
    echo "✅ Folder already exists: $USER_DB_DIR"
fi

# Export environment variables
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_DIR"
export MIOPEN_USER_DB_PATH="$USER_DB_DIR"

echo "🌱 MIOpen environment configured:"
echo "  MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR"
echo "  MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"

# --- SLURM Job Request ---
echo
echo "🚀 Requesting an interactive session on a GPU node..."
echo "   Partition: ${PARTITION}"
echo "   GPU: ${GRES}"
echo "   Time: ${TIME}"

# Launch the interactive session
srun --partition="$PARTITION" --gres="$GRES" --time="$TIME" --pty "$SHELL_TYPE"

# --- Notes ---
# Once connected, you'll be on the compute node.
# Run 'source /path/to/your/venv/bin/activate' to activate your environment.
# Type 'exit' to end the session and release the node.
