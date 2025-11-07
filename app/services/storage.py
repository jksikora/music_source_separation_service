import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AudioData:
    filename: str
    waveform: object
    sample_rate: int
    #created_at: float = time.time() - Future: Add timer for files expiration logic.  

class StorageInterface(ABC):
    @abstractmethod
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
    
    
class Storage(StorageInterface):
    def __init__(self):
        self._storage: dict[str, AudioData] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, file_id: str, audio_data: AudioData) -> None:
        async with self._lock:
            self._storage[file_id] = audio_data
            print(f"Audio file {audio_data.filename} with file id {file_id} successfuly saved")
    
    async def get(self, file_id: str) -> AudioData | None:
        async with self._lock:
            print("Getting:", file_id)
            return self._storage.get(file_id)
    
    async def delete(self, file_id: str) -> None:
        async with self._lock:
            self._storage.pop(file_id, None)

    async def clear(self) -> None:
        async with self._lock:
            self._storage.clear()

    async def exists(self, file_id: str) -> bool:
        async with self._lock:
            return file_id in self._storage

    async def list_ids(self) -> list[str]:
        async with self._lock:
            return list(self._storage.keys())

storage = Storage()