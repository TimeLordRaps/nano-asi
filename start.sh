#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Source Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate nano-asi

# Function to start Open WebUI
start_open_webui() {
    echo "Starting Open WebUI..."
    # Replace this with the actual command to start Open WebUI if different
    open-webui &
}

# Function to start Pipelines
start_pipelines() {
    echo "Starting Pipelines..."
    # Replace 'your_pipelines_app:app' with the actual module and application instance for Pipelines
    uvicorn your_pipelines_app:app --host 0.0.0.0 --port 9099
}

# Start both services
start_open_webui
start_pipelines

# Wait for any process to exit
wait -n

# Exit with the status of the first process to exit
exit $?
