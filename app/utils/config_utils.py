from app.schemas.worker_schemas import WorkerConfig
from pathlib import Path
import yaml

# === Load Worker Configuration Function ===
def load_worker_config(config_filename: str = "scnet01_config.yaml") -> WorkerConfig:
    """Function to check if the YAML configuration file exists and load it"""
    config_path = Path(__file__).resolve().parents[1] / "workers" / config_filename # Path to worker config file
    if not config_path.exists():
        raise FileNotFoundError(f"Missing worker configuration file at {config_path}") # Raise error if config file does not exist

    with open(config_path, "r", encoding="utf-8") as f: # Open SCNet worker config file using UTF-8 encoding for compatibility with different systems
        data = yaml.safe_load(f) or {} # Load YAML content safely, return empty dict if file is empty so the check below works correctly

    required_fields = ("worker_id", "model_type", "worker_address", "main_address") # Define required fields for worker configuration
    missing = [k for k in required_fields if not data.get(k)] # Check for missing or empty required fields
    if missing:
        raise ValueError(f"Missing or empty fields ({', '.join(missing)}) in {config_path}") # Raise error if any required field is missing or empty

    return WorkerConfig.model_validate(data) # Validate and return WorkerConfig instance using pydantic model validation because the input data is from an external source