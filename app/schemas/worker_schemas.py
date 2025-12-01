from pydantic import BaseModel

class Worker(BaseModel): # Information about each registered worker
    worker_id: str
    model_type: str
    worker_address: str
    
class WorkerData(Worker):
    status: str = "ready"  # e.g. "ready", "busy";

class WorkerConfig(Worker): 
    main_address: str