from pydantic import BaseModel

class AudioData(BaseModel): # Information about uploaded audio file
    filename: str
    waveform: object
    sample_rate: int
#   created_at: float = time.time() - Future: Add timer for files expiration logic.  

class StemData(BaseModel): # Information about each separated stem
    file_id: str
    filename: str
    download_url: str