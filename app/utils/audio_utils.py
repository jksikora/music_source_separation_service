from fastapi import File, UploadFile, HTTPException, Depends, Response
from app.services.storage import storage
from app.services.session_manager import session_manager
from app.schemas.audio_schemas import AudioEntry
from app.schemas.worker_schemas import WorkerConfig
from app.utils.logging_utils import get_logger
from pathlib import Path
import soundfile as sf
import io, torchaudio, httpx, zipfile, uuid

logger = get_logger(__name__) # Logger for audio_utils module


# === Verify Uploaded Audio File and Load Waveform Function ===
async def _audio_file_verification(file: UploadFile = File(...)) -> dict:
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


# === Get Inference Result from SCNet Worker Function ===
async def _get_result(worker: WorkerConfig, waveform: object, sample_rate: int, filename: str) -> tuple[io.BytesIO, dict, str, str]:
    audio_buffer = io.BytesIO() # Create in-memory buffer for audio data
    sf.write(audio_buffer, waveform.T, sample_rate, format="WAV") # Write waveform to buffer in WAV format
    audio_buffer.seek(0) # Reset buffer pointer to the beginning

    async with httpx.AsyncClient(timeout=None) as client:  # HTTP client with no timeout
        files = {"file": (filename, audio_buffer, "audio/wav")}  # Prepare file for upload
        response = await client.post(f"http://{worker.worker_address}/{worker.worker_id}/inference", files=files)  # Send inference request to SCNet worker
    if response.status_code == 200:
        logger.info(action="inference_request", status="success", data={"filename": filename, "worker_id": worker.worker_id, "address": worker.worker_address})
    else:
        logger.error(action="inference_request", status="failed", data={"worker_id": worker.worker_id, "status_code": response.status_code, "error": response.text})
        raise HTTPException(status_code=response.status_code, detail=response.text)

    headers = response.headers # Extract separation timestamps (if provided) from worker response headers
    t0_model = headers.get("separation-start") # Get separation timestamps from worker response headers
    t1_model = headers.get("separation-end")

    return io.BytesIO(response.content), t0_model, t1_model # Return in-memory ZIP buffer for received ZIP file and separation timestamps


# === Process Inference Result from SCNet Worker Function ===
async def _process_result(result_zip: io.BytesIO, filename: str) -> dict[str, AudioEntry]:
    with zipfile.ZipFile(result_zip, "r") as zip:  # Open ZIP file from buffer
            file_id = str(uuid.uuid4())  # Generate unique ID for the original file
            result = {}  # Dictionary to hold separation results
            entries = ', '.join(zip.namelist()) or 'none' # Prepare entries for logging; handles empty ZIP case
            logger.info(action="inference_reception", status="success", data={"filename": filename, "file_id": file_id, "entries": entries})

            for name in zip.namelist(): # Iterate over files in the ZIP
                with zip.open(name) as f: # Open each file in the ZIP
                    audio_bytes = f.read() # Read audio file bytes
                    buffer = io.BytesIO(audio_bytes) # Create in-memory buffer for audio data
                    output_waveform, output_sample_rate = sf.read(buffer, dtype="float32") # Read audio data from buffer

                    stem_name = Path(name).stem  # Get stem name without extension
                    stem_file_id = f"{stem_name}_{file_id}"  # Unique file ID for the stem
                    stem_filename = f"{stem_name}_{filename}"  # Filename for the stem

                    await storage.save(stem_file_id, stem_filename, output_waveform, output_sample_rate)  # Save stem audio data in storage
                    logger.info(action="audio_save", status="success", data={"file_id": stem_file_id, "filename": stem_filename})

                    result[stem_name] = AudioEntry(  # Store separation result for the stem
                        file_id=stem_file_id,
                        filename=stem_filename,
                        download_url=f"/download_audio/{stem_file_id}"
                    )

            return result


# === Perform Music Source Separation Using Available SCNet Worker Function ===
async def music_source_separation(audiofile: dict = Depends(_audio_file_verification), response: Response = None) -> dict[str, AudioEntry]:
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
        result_zip, t0_model, t1_model = await _get_result(worker, waveform, sample_rate, file.filename)
        result = await _process_result(result_zip, file.filename)  # Process the received ZIP file
        if not result or not result_zip:
            logger.warning(action="separation_completion", status="failed", data={"filename": file.filename, "error": "no_stems_extracted"})
            raise HTTPException(status_code=500, detail="No stems extracted from the audio file")

        if response is not None and (t0_model is not None and t1_model is not None):
            response.headers["separation-start"] = t0_model # Propagate separation timestamps to response headers (for HTTP client to measure separation-only time)
            response.headers["separation-end"] = t1_model
            logger.info(action="inference_timestamp_propagation", status="success", data={"filename": file.filename, "separation-start": t0_model, "separation-end": t1_model})

        logger.info(action="separation_completion", status="success", data={"filename": file.filename, "num_stems": len(result)})

        return result
        
    finally:
        await session_manager.release_worker(worker.worker_id) # Release the worker back to the session manager
        logger.info(action="worker_release", status="success", data={"worker_id": worker.worker_id, "model_type": worker.model_type})