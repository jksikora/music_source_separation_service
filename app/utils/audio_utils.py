from fastapi import File, UploadFile, HTTPException, Depends
from app.services.storage import storage
from app.services.session_manager import session_manager
from app.schemas.audio_schemas import AudioData, StemData
from app.utils.logging_utils import get_logger
from pathlib import Path
import soundfile as sf
from typing import Generator
import io, torchaudio, httpx, zipfile, uuid

logger = get_logger(__name__) # Logger for audio_utils module


# === Verify Uploaded Audio File and Load Waveform Function ===
async def audio_file_verification(file: UploadFile = File(...)) -> dict:
    """Function to verify if the uploaded file is a valid audio file and load its waveform"""
    try:
        audio_bytes = await file.read() # Read uploaded audio file bytes
        audio_buffer = io.BytesIO(audio_bytes) # Create in-memory buffer for audio data
        waveform, sample_rate = torchaudio.load(audio_buffer) # Checking if the data is a valid audio file
        logger.info(action="audio_loading", status="success", data={"filename": file.filename, "sample_rate": sample_rate, "shape": tuple(waveform.shape)})
    
    except Exception as e:
        logger.warning(action="audio_loading", status="failed", data={"filename": file.filename, "status_code": 400, "error": str(e)})
        raise HTTPException(status_code=400, detail="Invalid audio file")  # Raise error if not valid
    
    return {
        "file": file, 
        "waveform": waveform, 
        "sample_rate": sample_rate
    }


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


# === Perform Music Source Separation Using Available SCNet Worker Function ===
async def music_source_separation(audiofile: dict = Depends(audio_file_verification)) -> dict[str, StemData]:
    """Function to perform music source separation using an available SCNet worker"""
    file = audiofile["file"]
    waveform = audiofile["waveform"]
    sample_rate = audiofile["sample_rate"]
    
    worker = await session_manager.get_worker("scnet")  # Acquire an available SCNet worker
    if not worker:
        logger.warning(action="worker_acquisition", status="failed", data={"status_code": 503, "error": "no_available_workers"})
        raise HTTPException(status_code=503, detail="No available SCNet workers")
    logger.info(action="worker_acquisition", status="success", data={"worker_id": worker.worker_id, "model_type": worker.model_type})
    
    try:
        audio_buffer = io.BytesIO() # Create in-memory buffer for audio data
        sf.write(audio_buffer, waveform.T, sample_rate, format="WAV") # Write waveform to buffer in WAV format
        audio_buffer.seek(0) # Reset buffer pointer to the beginning

        async with httpx.AsyncClient(timeout=None) as client:  # HTTP client with no timeout
            files = {"file": (file.filename, audio_buffer, "audio/wav")}  # Prepare file for upload
            response = await client.post(f"http://{worker.worker_address}/{worker.worker_id}/inference", files=files)  # Send inference request to SCNet worker
        if response.status_code == 200:
            logger.info(action="inference_request", status="success", data={"filename": file.filename, "worker_id": worker.worker_id, "address": worker.worker_address})
        else:
            logger.error(action="inference_request", status="failed", data={"worker_id": worker.worker_id, "status_code": response.status_code, "error": response.text})
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result_zip = io.BytesIO(response.content)  # In-memory buffer for received ZIP file
        with zipfile.ZipFile(result_zip, "r") as zip:  # Open ZIP file from buffer
            file_id = str(uuid.uuid4())  # Generate unique ID for the original file
            result = {}  # Dictionary to hold separation results
            entries = ', '.join(zip.namelist()) or 'none' # Prepare entries for logging; handles empty ZIP case
            logger.info(action="inference_reception", status="success", data={"filename": file.filename, "file_id": file_id, "entries": entries})

            for name in zip.namelist(): # Iterate over files in the ZIP
                with zip.open(name) as f: # Open each file in the ZIP
                    audio_bytes = f.read() # Read audio file bytes
                    buffer = io.BytesIO(audio_bytes) # Create in-memory buffer for audio data
                    output_waveform, output_sample_rate = sf.read(buffer, dtype="float32") # Read audio data from buffer

                    stem_name = Path(name).stem # Get stem name without extension
                    stem_file_id = f"{stem_name}_{file_id}" # Unique file ID for the stem
                    filename = f"{stem_name}_{file.filename}" # Filename for the stem

                    audio_data = AudioData( # Create AudioData instance for the stem
                        filename=filename,
                        waveform=output_waveform,
                        sample_rate=output_sample_rate
                    )

                    await storage.save(stem_file_id, audio_data) # Save stem audio data in storage
                    logger.info(action="audio_save", status="success", data={"file_id": stem_file_id, "filename": filename})

                    result[stem_name] = StemData( # Store separation result for the stem
                        file_id=stem_file_id,
                        filename=filename,
                        download_url=f"/download_audio/{stem_file_id}"
                    )

        logger.info(action="separation_completion", status="success", data={"filename": file.filename, "num_stems": len(result)})
        return result
        
    finally:
        await session_manager.release_worker(worker.worker_id) # Release the worker back to the session manager
        logger.info(action="worker_release", status="success", data={"worker_id": worker.worker_id, "model_type": worker.model_type})