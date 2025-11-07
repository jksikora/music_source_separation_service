from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.services import (
    convert_to_audio_buffer,
    buffer_generator,
    music_source_separation
)
from app.services import storage, session_manager
from app.schemas import SeparationResult

router = APIRouter()

# --- Upload audio Endpoint ---

@router.post("/upload_audio", response_model = SeparationResult)
async def upload_audio(result: dict = Depends(music_source_separation)):
    return {"result": result}

# --- Download audio Endpoint ---

@router.get("/download_audio/{file_id}")
async def download_audio(file_id: str):
    if not await storage.exists(file_id):
        raise HTTPException(status_code=404, detail="File not found") #Error if file ID not found
    
    audio_data = await storage.get(file_id) #Retrieve the audio data from memory
    if audio_data is None:
        raise HTTPException(status_code=404, detail="File not found")

    filename = audio_data.filename
    waveform = audio_data.waveform
    sample_rate = audio_data.sample_rate 
    
    save_buffer, size, filename = convert_to_audio_buffer(waveform, sample_rate, filename) #Convert audio data to buffer for download
    
    headers = { #Set response headers for file download
        "Content-Disposition": f'attachment; filename="{filename}"', 
        "Content-Length": str(size)
        #Optional "Cache-Control": "public, max-age=86400" - Cache for 1 day, helpful for repeated downloads
    }

    return StreamingResponse(buffer_generator(save_buffer), media_type="audio/wav", headers=headers) #Stream the audio file as a response
