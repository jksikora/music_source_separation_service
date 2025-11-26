from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from scnet.inference import Seperator, SCNet
from app.utils.config_utils import load_worker_config
from app.utils.worker_utils import validate_outputs, zipstream_generator
from app.utils.logging_utils import setup_logging, get_logger
from ml_collections import ConfigDict
from pathlib import Path
import soundfile as sf
import yaml, asyncio, importlib.util, io, httpx 

app = FastAPI(title = "SCNetWorker")

setup_logging() # Setup logging configuration
logger = get_logger(__name__) # Logger for SCNet Worker


# === Global Separator Instance and Inference Lock ===
separator: Seperator | None = None # Initialize separator instance on startup
inference_lock = asyncio.Lock() # Initialize lock on startup to serialize inference requests


# === Worker Configuration Loaded from YAML File ===
try:
    worker_data = load_worker_config()
except (FileNotFoundError, ValueError) as exc:
    logger.error(action="config_loading", status="failed", data={"error": "invalid_worker_data", "details": str(exc)})
    raise RuntimeError("Loading worker configuration failed") from exc # Raise error if config file is missing or invalid, from exc chains the original exception to the new one

worker_id = worker_data.worker_id
model_type = worker_data.model_type
main_address = worker_data.main_address
worker_address = worker_data.worker_address

# === Startup Event ===
@app.on_event("startup")
async def load_model() -> None:
    """On startup load SCNet model, initialize separator instance on startup and try registering worker"""
    global separator
    
    spec = importlib.util.find_spec("scnet") # Check if SCNet package is importable
    if spec is None:
        raise ImportError("SCNet package not found")
    
    scnet_root = Path(spec.submodule_search_locations[0]).resolve().parent # Get SCNet package root
    config_path = str(scnet_root / "conf" / "config.yaml") # Path to SCNet default config
    project_root = Path(__file__).resolve().parents[2] # Get the project root
    checkpoint_path = str(project_root / "checkpoints" / "scnet" / "checkpoint.th") # Path to SCNet checkpoint

    with open(config_path, "r") as f: # Load SCNet config file
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader)) # Load YAML content
        
    model = SCNet(**config.model) # Unpack model configuration and create SCNet model instance
    model.eval() # Set model to evaluation mode
    separator = Seperator(model, checkpoint_path) # Load model checkpoint into Seperator instance, select device automatically (CPU/GPU) and prepare for inference

    logger.info(action="model_loading", status="success", data={"worker_id": worker_id, "model_type": model_type})
    await try_register()


# === Inference Endpoint ===
@app.post(f"/{worker_id}/inference")
async def infer(file: UploadFile) -> StreamingResponse:
    """Endpoint for performing music source separation inference on uploaded audio file"""
    try:
        if separator is None: # Check if separator is initialized
            logger.error(action="inference", status="failed", data={"worker_id": worker_id, "filename": file.filename, "error": "model_not_loaded"})
            raise HTTPException(status_code=503, detail="Model not loaded")

        audio = await file.read() # Read uploaded audio file
        audio_buffer = io.BytesIO(audio) # Create in-memory buffer for audio data
        waveform, sample_rate = sf.read(audio_buffer, dtype="float32") # Read audio data from buffer
          
        async with inference_lock: # Acquire lock to serialize inference requests
            output_waveforms, output_sample_rates = await asyncio.to_thread(separator.separate_music_file, waveform, sample_rate) # Perform inference in a separate thread

        validate_outputs(output_waveforms, output_sample_rates, worker_id=worker_id, filename=file.filename) # Validate inference outputs

        zipstream, headers = zipstream_generator(output_waveforms, output_sample_rates, worker_id, file.filename) # Create streaming ZIP response
        return StreamingResponse(zipstream, media_type="application/zip", headers=headers) #Stream the ZIP file as a response
    
    except HTTPException: 
        raise # Re-raise HTTP exceptions from _validate_outputs to preserve status codes

    except Exception as e:
        logger.exception(action="inference", status="failed", data={"worker_id": worker_id, "filename": file.filename, "status_code": 500, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


# === Register Request Endpoint ===
@app.post("/register_request")
async def register_request() -> None:
    """Endpoint for main app to request worker registration in Session Manager"""
    try:
        await try_register()
        logger.info(action="registration_request", status="success", data={"worker_id": worker_id})

    except Exception:
        logger.warning(action="registration_request", status="failed", data={"worker_id": worker_id, "status_code": 500, "error": "registration_failed"})
        raise HTTPException(status_code=500, detail="Registration attempt failed")


# === Try Register SCNet Worker Function ===
async def try_register() -> None:
    """Function to attempt registering SCNet worker with main app in Session Manager if model is initialized"""
    if separator is None:
        logger.warning(action="registration_attempt", status="failed", data={"worker_id": worker_id, "status_code": 500, "error": "model_not_initialized"})
        return  # skip the HTTP call until load_model completes

    worker_data = {
        "worker_id": worker_id,
        "model_type": model_type,
        "worker_address": worker_address,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client: # HTTP client with timeout
            response = await client.post(f"http://{main_address}/register_worker", json=worker_data) # Send registration request to main app
        if response.status_code == 200:
            logger.info(action="registration_attempt", status="success", data={"worker_id": worker_id, "model_type": model_type, "address": worker_address})
        else:
            logger.warning(action="registration_attempt", status="failed", data={"worker_id": worker_id, "status_code": response.status_code, "error": response.text})
    
    except Exception as e:
        logger.warning(action="registration_attempt", status="failed", data={"worker_id": worker_id, "status_code": 500, "error": str(e)})