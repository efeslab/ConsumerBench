import os
import time
import subprocess
import sys

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import src.globals as globals


def util_run_server_script_check_log(script_path: str, server_dir: str, stdout_log_path: str, stderr_log_path: str, stderr_ready_patterns,
                              stdout_ready_patterns, listen_port, api_port, model, device="gpu", mps=100):
    """Run a script and check log files for startup indicators"""
    server_pid = -1
    max_wait = 120  # Maximum seconds to wait
    log_dir = os.path.join(globals.get_results_dir(), "server_logs")
    os.makedirs(log_dir, exist_ok=True)

    stdout_log = os.path.join(log_dir, f"{stdout_log_path}_{api_port}.log")
    stderr_log = os.path.join(log_dir, f"{stderr_log_path}_{api_port}.log")

    # Start the server process with log file redirection
    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        process = subprocess.Popen(
            [script_path, str(server_dir), str(listen_port), str(api_port), str(model), str(device), str(mps)],
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,  # Important for server processes
        )

    start_time = time.time()
    # Define patterns that indicate successful startup
    found_patterns = {}
    for pattern in stderr_ready_patterns:
        found_patterns[pattern] = False
    for pattern in stdout_ready_patterns:
        found_patterns[pattern] = False

    def check_logs():
        """Check log files for startup indicators"""
        nonlocal server_pid
        
        # Check stderr log
        for log in [stderr_log, stdout_log]:
            try:
                with open(log, 'r') as f:
                    content = f.read()
                    for pattern in stderr_ready_patterns:
                        if pattern in content and not found_patterns[pattern]:
                            found_patterns[pattern] = True
                            print(f"Server indicator found: {pattern}")
                    
                    for pattern in stdout_ready_patterns:
                        if pattern in content and not found_patterns[pattern]:
                            found_patterns[pattern] = True
                            print(f"Server indicator found: {pattern}")
                            
            except Exception as e:
                print(f"Error reading stderr log: {e}")
            
    # Wait for server to be ready using both methods
    while time.time() - start_time < max_wait:
        # Check log files for indicators
        check_logs()
        
        # If we found all log indicators, check if server is responding
        if all(found_patterns.values()):
            print("All server startup indicators found in logs")
            break
                    
        # Wait before checking again
        time.sleep(1)
    
    # Check if we timed out
    if time.time() - start_time >= max_wait:
        print("WARNING: Timed out waiting for server to be ready")
        # You might want to kill the process here
    
    return server_pid

# [ are we using this?]
def parse_commands(filename: str):
    """Read commands from a file"""
    commands: list[str] = []

    with open(filename, 'r') as f:
        current_command = """"""
        for line in f:
            if line == "\n":
                continue

            if line.startswith("[COMMAND]"):
                if current_command != """""":
                    commands.append(current_command)
                    current_command = """"""
                    
                continue
            else:
                # remove last \n from line
                line = line.rstrip("\n\\")
                current_command += line

        # print(current_command)
        if current_command != """""":
            commands.append(str(current_command))

    return commands