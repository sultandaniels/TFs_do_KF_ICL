import subprocess
import argparse

def run_commands(commands):
    for command in commands:
        try:
            print(f"Running command: {command}")
            result = subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}\n{e}")
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
        except KeyboardInterrupt:
            print("Process interrupted by user")
            break

if __name__ == "__main__":
    
    # Example commands with string formatting
    commands = [
        "cd ../transformers",
        "pip install -e .",
        "cd ../TFs_do_KF_ICL/src",
        "conda init",
        "conda activate mop_arm64",
        "pip install pytorch_lightning dimarray",
        "pip install -U tensorboard tensorboardX"
    ]
    
    # Run the commands
    run_commands(commands)