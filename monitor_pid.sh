#!/bin/bash

# Replace <PID> with the actual PID of the process you want to monitor
PID=3822243
OUTPUT_FILE="nvidia_smi_snapshot.txt"

# Function to capture nvidia-smi output
capture_nvidia_smi() {
    nvidia-smi > $OUTPUT_FILE
    echo "nvidia-smi snapshot saved to $OUTPUT_FILE"
}

# Monitor the process
while kill -0 $PID 2> /dev/null; do
    sleep 1
done

# Capture nvidia-smi output when the process terminates
capture_nvidia_smi