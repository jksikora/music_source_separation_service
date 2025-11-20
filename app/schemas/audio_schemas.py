from pydantic import BaseModel, Field
import time

class AudioData(BaseModel): # Information about uploaded audio file
    filename: str
    waveform: object
    sample_rate: int
    created_at: float = Field(default_factory=time.time) # A default value is set to the time when the model is constructed

class StemData(BaseModel): # Information about each separated stem
    file_id: str
    filename: str
    download_url: str