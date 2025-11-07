import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class WorkerData:
    worker_id: str 
    model_type: str
    address: str
    status: str = "ready"

class SessionManager:
    def __init__(self):
        self._workers: dict[str, WorkerData] = {}
        self._lock = asyncio.Lock()
    
    async def register_worker(self, worker_id: str, model_type: str, address: str) -> None:
        async with self._lock:
            self._workers[worker_id] = WorkerData(worker_id = worker_id, model_type = model_type, address = address)
            print(f"Worker '{worker_id}' for model '{model_type}' successfuly registered at {address}")
    
    async def get_worker(self, model_type: str) -> WorkerData | None:
        async with self._lock:
            for worker in self._workers.values():
                if worker.model_type == model_type and worker.status == "ready":
                    worker.status = "busy"
                    print(f"Worker {worker.worker_id} for model {worker.model_type} successfuly acquired")
                    return worker
            return None
    
    async def release_worker(self, worker_id: str) -> None:
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = "ready"
                print(f"Worker {self._workers[worker_id].worker_id} for model {self._workers[worker_id].model_type} successfuly released")
    
    async def list_workers(self) -> list[WorkerData]:
        async with self._lock:
            return list(self._workers.values())

    async def clear(self) -> None:
        async with self._lock:
            self._workers.clear()

session_manager = SessionManager()