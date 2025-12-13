from app.schemas.worker_schemas import WorkerData
from app.utils.logging_utils import get_logger
import asyncio

# === Session Manager Class ===
class SessionManager:
    """Class to manage worker registration and allocation for music source separation tasks"""
    def __init__(self):
        self._workers: dict[str, WorkerData] = {} # In-memory dictionary to hold registered workers
        self._lock = asyncio.Lock() # Async lock for thread-safe operations
        self._logger = get_logger(__name__) # Logger for session manager

    async def register_worker(self, worker_id: str, model_type: str, worker_address: str) -> None:
        """Function to register a new worker in storage"""
        async with self._lock: # Acquire lock for thread-safe operation
            if worker_id in self._workers: # Check if worker already registered (prevent duplicates)
                self._logger.warning(action="worker_registration", status="failed", data={"worker_id": worker_id, "model_type": model_type, "worker_address": worker_address, "error": "worker_already_registered"})
                return
            
            self._workers[worker_id] = WorkerData(worker_id = worker_id, model_type = model_type, worker_address = worker_address) # Add worker to storage using defined structure
            self._logger.info(action="worker_registration", status="success", data={"worker_id": worker_id, "model_type": model_type, "worker_address": worker_address})

    async def get_worker(self, model_type: str) -> WorkerData | None:
        """Function to acquire an available worker from storage"""
        async with self._lock:
            for worker in self._workers.values():
                if worker.model_type == model_type and worker.status == "ready":
                    worker.status = "busy"
                    self._logger.info(action="worker_acquisition", status="in progress", data={"worker_id": worker.worker_id, "model_type": worker.model_type})
                    return worker
            return None
    
    async def release_worker(self, worker_id: str) -> None:
        """Function to release a worker after task completion in storage"""
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = "ready"
                self._logger.info(action="worker_release", status="in progress", data={"worker_id": self._workers[worker_id].worker_id, "model_type": self._workers[worker_id].model_type})
    
    async def list_workers(self) -> list[WorkerData]:
        """Function to list all registered workers in storage"""
        async with self._lock:
            worker_ids = ', '.join(self._workers.keys()) or 'none'
            self._logger.info(action="list_read", status="success", data={"count": len(self._workers), "ids": worker_ids})
            return list(self._workers.values())

    async def clear(self) -> None:
        """Function to clear all registered workers in storage"""
        async with self._lock:
            previous_count = len(self._workers)
            self._workers.clear()
            self._logger.info(action="list_clear", status="success", data={"previous_count": previous_count})


session_manager = SessionManager() # Global session manager instance