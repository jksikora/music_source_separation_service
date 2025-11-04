from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
import torchaudio
import io
import uuid
import soundfile as sf

app=FastAPI()

storage = {}

# Audio file verification function
async def audiofile_verification(file: UploadFile = File(...)) -> dict:
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

# Music source separation function
def music_source_separation(audiofile: dict = Depends(audiofile_verification)) -> dict:
    file = audiofile["file"]
    waveform = audiofile["waveform"]
    sample_rate = audiofile["sample_rate"]
    
    file_id = str(uuid.uuid4()) #Generate unique ID for the original file

    stems = { #A simulation of what real models would output
        "vocals": waveform,  
        "drums": waveform,
        "bass": waveform,
        "other": waveform
    }

    result = {}
    for stem_name, stem_waveform in stems.items():
        stem_file_id = f"{stem_name}_{file_id}" #Generate ID for each stem
        storage[stem_file_id] = (f"{stem_name}_{file.filename}", stem_waveform, sample_rate) #Store each stem in memory
        result[stem_name] = { #Using the stem_name as a key = Creating a smaller dict inside the big result dict for each stem
            "file_id": stem_file_id,
            "filename": f"{stem_name}_{file.filename}",
            "audio": stem_waveform,
            "sample_rate": sample_rate,
            "download_url": f"/download_audio/{stem_file_id}"
        }

    return result

# Function to convert torch.Tensor (waveform) to audio buffer
def convert_to_audio_buffer(waveform, sample_rate: int, filename: str) -> io.BytesIO:
    #Optional_1: Convert to WAV format before sending <- this is chosen approach here
    #Optional_2: If download is too slow, consider making temporary files instead of in-memory buffers, instead of using io.BytesIO() use aiofiles or similar libraries for async file handling
    save_buffer = io.BytesIO() #Buffer to save the audio data, NOT ASYNC I/O
    sf.write(save_buffer, waveform.T.numpy(), sample_rate, format="WAV") #Save audio data to buffer in WAV format; Optional, subtype="PCM_16"
    save_buffer.seek(0) #Reset buffer pointer to the beginning

    size = save_buffer.getbuffer().nbytes #Get size of the buffer
    download_filename = filename if filename.lower().endswith(".wav") else f"{filename}.wav" #Ensure filename ends with .wav

    return save_buffer, size, download_filename

# Generator to control the chunk size for streaming
chunk_size = 256 * 1024  # 256 KB
def buffer_generator(buffer: io.BytesIO, chunk_size: int = chunk_size):
    buffer.seek(0)
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        yield chunk

# Middleware for error handling
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return response

# Endpoint to upload audio file
@app.post("/upload_audio")
async def upload_audio(result: dict = Depends(music_source_separation)):
    return {"message": "Audio file uploaded successfully!", "result": result}

# Endpoint to download audio file
@app.get("/download_audio/{file_id}")
async def download_audio(file_id: str):
    if file_id not in storage:
        raise HTTPException(status_code=404, detail="File not found") #Error if file ID not found
    
    filename, waveform, sample_rate = storage[file_id] #Retrieve the audio data from memory
    save_buffer, size, filename = convert_to_audio_buffer(waveform, sample_rate, filename) #Convert audio data to buffer for download
    
    headers = { #Set response headers for file download
        "Content-Disposition": f'attachment; filename="{filename}"', 
        "Content-Length": str(size)
        #Optional "Cache-Control": "public, max-age=86400" - Cache for 1 day, helpful for repeated downloads
    }

    return StreamingResponse(buffer_generator(save_buffer), media_type="audio/wav", headers=headers) #Stream the audio file as a response