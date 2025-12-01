from app.utils.logging_utils import get_logger
import soundfile as sf
from typing import Generator
import io

logger = get_logger(__name__) # Logger for streaming_utils module


# === Convert torch.Tensor (waveform) to an Audio Buffer ===
def convert_to_audio_buffer(waveform: object, sample_rate: int, filename: str) -> tuple[io.BytesIO, int, str]:
    """Function to convert a waveform tensor to an audio buffer in WAV format for streaming purpose"""
    save_buffer = io.BytesIO() # Buffer to save the audio data, NOT ASYNC I/O
    sf.write(save_buffer, waveform, sample_rate, format="WAV") # Save audio data to buffer in WAV format; Optional, subtype="PCM_16"
    save_buffer.seek(0) # Reset buffer pointer to the beginning
    size = save_buffer.getbuffer().nbytes # Get size of the buffer

    download_filename = filename if filename.lower().endswith(".wav") else f"{filename}.wav" # Ensure filename ends with .wav
    logger.debug(action="buffer_conversion", status="success", data={"filename": download_filename, "size": size, "sample_rate": sample_rate})
    return save_buffer, size, download_filename


# === Stream Buffer in Specified Size Chunks Function ===
def buffer_generator(buffer: io.BytesIO, chunk_size: int = 256 * 1024) -> Generator[bytes, None, None]:  # 256 KB
    """Function to yield chunks of data from a buffer for streaming in download endpoint"""
    buffer.seek(0)  # Reset buffer pointer to the beginning
    total = 0
    while True:
        chunk = buffer.read(chunk_size) # Read chunk of data
        if not chunk:
            break
        total += len(chunk)
        yield chunk
    logger.debug(action="buffer_streaming", status="success", data={"total_size": total})