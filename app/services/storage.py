from app.schemas.audio_schemas import AudioData
from app.utils.logging_utils import get_logger
from abc import ABC, abstractmethod
import asyncio

# === Abstract Storage Interface ===
class StorageInterface(ABC): # Abstract Base Class for storage implementations
    @abstractmethod # Decorator to indicate abstract method
    async def save(self, file_id: str, audio_data: AudioData) -> None:
        raise NotImplementedError
    
    @abstractmethod
    async def get(self, file_id: str) -> AudioData | None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, file_id: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    async def exists(self, file_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def list_ids(self) -> list[str]:
        raise NotImplementedError
    

# === Storage Class ===
class Storage(StorageInterface):
    """Class with concrete implementation of StorageInterface to manage audio data using in-memory dictionary"""
    def __init__(self):
        self._storage: dict[str, AudioData] = {} # In-memory storage dictionary
        self._lock = asyncio.Lock() # Async lock for thread-safe operations
        self._logger = get_logger(__name__) # Logger for storage
    
    async def save(self, file_id: str, audio_data: AudioData) -> None:
        """Function to save audio data in storage"""
        async with self._lock: # Acquire lock for thread-safe operation
            self._storage[file_id] = audio_data 
            self._logger.info(action="audio_save", status="in progress", data={"file_id": file_id, "filename": audio_data.filename}) 
    
    async def get(self, file_id: str) -> AudioData | None:
        """Function to get audio data from storage"""
        async with self._lock:
            self._logger.info(action="audio_get", status="in progress", data={"file_id": file_id})
            return self._storage.get(file_id)
    
    async def delete(self, file_id: str) -> None:
        """Function to delete audio data from storage"""
        async with self._lock:
            self._storage.pop(file_id, None)
            self._logger.info(action="audio_delete", status="in progress", data={"file_id": file_id})

    async def exists(self, file_id: str) -> bool:
        """Function to check if audio data exists in storage"""
        async with self._lock:
            exists = file_id in self._storage
            self._logger.debug(action="audio_exists", status="success", data={"file_id": file_id, "exists": exists})
            return exists

    async def list_ids(self) -> list[str]:
        """Function to list all audio file IDs in storage"""
        async with self._lock:
            file_ids = list(self._storage.keys())
            ids = ', '.join(file_ids) or 'none'
            self._logger.debug(action="list_read", status="success", data={"count": len(file_ids), "ids": ids})
            return file_ids
    
    async def clear(self) -> None:
        """Function to clear all audio data from storage"""
        async with self._lock:
            previous_count = len(self._storage)
            self._storage.clear()
            self._logger.info(action="list_clear", status="success", data={"previous_count": previous_count})


storage = Storage() # Global storage instance