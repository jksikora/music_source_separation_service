from pydantic import BaseModel

class WorkerData(BaseModel): # Information about each registered worker
    worker_id: str
    model_type: str
    worker_address: str
    main_address: str
    status: str = "ready"  # e.g., "ready", "busy"; when created, status is "ready"