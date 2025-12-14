from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.utils.audio_utils import music_source_separation
from app.utils.streaming_utils import convert_to_audio_buffer, buffer_generator
from app.utils.logging_utils import get_logger
from app.services.storage import storage
from app.schemas.audio_schemas import AudioEntry

audio_router = APIRouter() # Create API router for audio routes
logger = get_logger(__name__) # Logger for audio_routes module


# === Upload Audio Endpoint ===
@audio_router.post("/upload_audio/{model}", response_model=dict[str, AudioEntry])
async def upload_audio(result: dict[str, AudioEntry] = Depends(music_source_separation)) -> dict[str, AudioEntry]:
    """Endpoint to upload audio file for music source separation; The response model enforces the contract returned by the dependency result"""
    return result


# === Download Audio Endpoint ===
@audio_router.get("/download_audio/{file_id}")
async def download_audio(file_id: str) -> StreamingResponse:
    """Endpoint to download processed audio file with file ID"""
    if not await storage.exists(file_id): # Check if file ID exists in storage
        logger.error(action="download_request", status="failed", data={"file_id": file_id, "status_code": 404, "error": "file_not_found"})
        raise HTTPException(status_code=404, detail="File not found") # Error if file ID not found
    
    audio_data = await storage.get(file_id) # Retrieve the audio data from memory
    filename = audio_data.filename
    waveform = audio_data.waveform
    sample_rate = audio_data.sample_rate 
    
    save_buffer, size, filename = convert_to_audio_buffer(waveform, sample_rate, filename) # Convert audio data to buffer for download
    
    headers = { # Set response headers for file download
        "Content-Disposition": f'attachment; filename="{filename}"', 
        "Content-Length": str(size)
        #Optional "Cache-Control": "public, max-age=86400" - Cache for 1 day, helpful for repeated downloads
    }

    logger.info(action="download_stream", status="success", data={"file_id": file_id, "filename": filename, "size": size})
    return StreamingResponse(buffer_generator(save_buffer), media_type="audio/wav", headers=headers) #Stream the audio file as a response