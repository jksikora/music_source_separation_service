from fastapi import File, UploadFile, HTTPException, Depends
import httpx
import io
import torchaudio
import soundfile as sf
import zipfile
import uuid
from pathlib import Path
from app.services import storage, AudioData
from app.services.session_manager import session_manager
from app.schemas import SeparationResult

# --- Audio file verification ---

async def audio_file_verification(file: UploadFile = File(...)) -> dict:
    try:
        audio_bytes = await file.read() 
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer) #Checking if the data is a valid audio file
    except Exception as e:
        print(f"Failed to load audio: {e}") 
        raise HTTPException(status_code=400, detail="Invalid audio file") #Error if not valid
    
    return {
        "file": file, 
        "waveform": waveform, 
        "sample_rate": sample_rate
    }

# --- Convert torch.Tensor (waveform) to an audio buffer ---

def convert_to_audio_buffer(waveform, sample_rate: int, filename: str) -> io.BytesIO:
    save_buffer = io.BytesIO() #Buffer to save the audio data, NOT ASYNC I/O
    sf.write(save_buffer, waveform, sample_rate, format="WAV") #Save audio data to buffer in WAV format; Optional, subtype="PCM_16"
    save_buffer.seek(0) #Reset buffer pointer to the beginning

    size = save_buffer.getbuffer().nbytes #Get size of the buffer
    download_filename = filename if filename.lower().endswith(".wav") else f"{filename}.wav" #Ensure filename ends with .wav

    return save_buffer, size, download_filename

# --- Stream buffer in specified size chunks ---

chunk_size = 256 * 1024  # 256 KB
def buffer_generator(buffer: io.BytesIO, chunk_size: int = chunk_size):
    buffer.seek(0)
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        yield chunk

# --- Temporarly Music Source Separation core ---

async def music_source_separation(audiofile: dict = Depends(audio_file_verification)) -> SeparationResult:
    file = audiofile["file"]
    waveform = audiofile["waveform"]
    sample_rate = audiofile["sample_rate"]
    
    worker = await session_manager.get_worker("scnet")
    if not worker:
        raise HTTPException(status_code=503, detail="No available SCNet workers")
    
    try:
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, waveform.T, sample_rate, format="WAV")
        audio_buffer.seek(0)

        async with httpx.AsyncClient(timeout=None) as client:
            files = {"file": (file.filename, audio_buffer, "audio/wav")}
            response = await client.post(f"http://{worker.address}/{worker.worker_id}/inference", files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result_zip = io.BytesIO(response.content)
        with zipfile.ZipFile(result_zip, "r") as zip:
            file_id = str(uuid.uuid4()) #Generate unique ID for the original file
            result = {} 

            for name in zip.namelist():
                with zip.open(name) as f:
                    audio_bytes = f.read()
                    buffer = io.BytesIO(audio_bytes)
                    output_waveform, output_sample_rate = sf.read(buffer, dtype="float32")

                    stem_name = Path(name).stem
                    stem_file_id = f"{stem_name}_{file_id}"
                    filename = f"{stem_name}_{file.filename}"

                    audio_data = AudioData(
                        filename=filename,
                        waveform=output_waveform,
                        sample_rate=output_sample_rate
                    )

                    await storage.save(stem_file_id, audio_data)

                    result[stem_name] = {
                        "file_id": stem_file_id,
                        "filename": filename,
                        "download_url": f"/download_audio/{stem_file_id}"
                    }

        return result
        
    finally:
        await session_manager.release_worker(worker.worker_id)