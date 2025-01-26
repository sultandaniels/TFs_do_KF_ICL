import subprocess
import argparse

def run_commands(commands):
    for command in commands:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error executing command: {command}\n{stderr.decode('utf-8')}")
        else:
            print(f"Output of command {command}:\n{stdout.decode('utf-8')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a series of terminal commands.")
    parser.add_argument("screen_name", type=str, help="the name of the screen to open")
    args = parser.parse_args()
    
    # Example commands with string formatting
    commands = [
        f"bash Miniforge3-Linux-aarch64.sh",
        f"source ~/.bashrc",
        ""
    ]
    
    # Run the commands
    run_commands(commands)