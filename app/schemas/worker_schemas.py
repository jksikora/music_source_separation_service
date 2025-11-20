from pydantic import BaseModel

class WorkerData(BaseModel): # Information about each registered worker
    worker_id: str
    model_type: str
    worker_address: str
    status: str = "ready"  # e.g., "ready", "busy"; when created, status is "ready"

class WorkerConfig(BaseModel): # Configuration for worker setup
    worker_id: str
    model_type: str
    app_address: str
    worker_address: str