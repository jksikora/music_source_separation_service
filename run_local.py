from app.utils.config_utils import load_worker_config
from app.utils.logging_utils import get_logger
import subprocess, sys

# === Helper Function to Extract Port From Config ===
def _get_port_from_address(address: str) -> int:
    """Extract port from address configured in config file"""
    if not address or ':' not in address:
        raise ValueError(f"Invalid address: {address}")
    return int(address.split(':')[1])


# === Main Function to Run All Services Locally ===
def main():
    scnet_cfg = load_worker_config("scnet", 1)
    scnet_port = _get_port_from_address(scnet_cfg.worker_address)
    dttnet_cfg = load_worker_config("dttnet", 1)
    dttnet_port = _get_port_from_address(dttnet_cfg.worker_address)
    main_port = _get_port_from_address(scnet_cfg.main_address)

    commands = [
        [sys.executable, '-m', 'uvicorn', 'app.main:app', '--reload', '--port', str(main_port)],
        [sys.executable, '-m', 'uvicorn', 'workers.dttnet.dttnet_worker:app', '--reload', '--port', str(dttnet_port)],
        [sys.executable, '-m', 'uvicorn', 'workers.scnet.scnet_worker:app', '--reload', '--port', str(scnet_port)],
    ]

    processes = [] # List to hold subprocesses
    try: # For graceful shutdown on Ctrl+C
        for cmd in commands:
            print(f"Launching: {' '.join(cmd)}")
            processes.append(subprocess.Popen(cmd)) # Start each service as a subprocess without blocking the code execution; Popen represents the running process and allows interaction with it
        print("All services started. Press Ctrl+C to stop.")
        for process in processes: # Wait for each subprocess to finish one by one
            process.wait() # Wait for each subprocess to complete (blocking); Python script stops here until process exit
    except KeyboardInterrupt: # Handle Ctrl+C gracefully
        for process in processes:
            process.terminate() # Send a termination signal to each subprocess on Ctrl+C 
        for process in processes:
            process.wait() # Wait for each subprocess to complete after termination


# === Entry Point ===
if __name__ == "__main__":
    main()