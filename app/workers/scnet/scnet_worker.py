from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.workers.scnet.scnet_model import SCNetModel
from contextlib import asynccontextmanager
from app.utils.config_utils import load_worker_config
from app.utils.worker_utils import validate_outputs, zipstream_generator
from app.utils.logging_utils import setup_logging, get_logger
import httpx 

# === Lifespan Event ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup load SCNet model and try registering worker"""
    global scnet_model
    scnet_model = SCNetModel()
    await scnet_model.load_model(worker_id, model_type) # Load SCNet model on startup
    await try_register()
    yield # Pauses here; Code after yield runs on shutdown


app = FastAPI(title = "SCNetWorker", lifespan=lifespan)

setup_logging() # Setup logging configuration
logger = get_logger(__name__) # Logger for SCNet Worker


# === Worker Configuration Loaded from YAML File ===
try:
    worker_config = load_worker_config("scnet", 1)
except (FileNotFoundError, ValueError) as exc:
    logger.error(action="config_loading", status="failed", data={"error": "invalid_worker_config", "details": str(exc)})
    raise RuntimeError("Loading worker configuration failed") from exc # Raise error if config file is missing or invalid, from exc chains the original exception to the new one

worker_id = worker_config.worker_id
model_type = worker_config.model_type
main_address = worker_config.main_address
worker_address = worker_config.worker_address


# === Inference Endpoint ===
@app.post(f"/{worker_id}/inference")
async def inference(file: UploadFile) -> StreamingResponse:
    """Endpoint for performing music source separation inference on uploaded audio file"""
    try:
        output_waveforms, output_sample_rates, t0_model, t1_model = await scnet_model.perform_inference(file, worker_id)  # Ensure model is loaded
        validate_outputs(output_waveforms, output_sample_rates, worker_id=worker_id, filename=file.filename) # Validate inference outputs
        zipstream, headers = zipstream_generator(output_waveforms, output_sample_rates, worker_id, file.filename) # Create streaming ZIP response
        logger.info(action="inference_processing", status="success", data={"worker_id": worker_id, "filename": file.filename, "num_stems": len(output_waveforms)})

        headers = dict(headers) if headers is not None else {} # If headers are provided (by zipstream_generator) make a shallow copy, else create empty dict
        headers.setdefault("separation-start", str(t0_model)) # Attach separation timestamps to response headers so caller can measure separation-only time; Add separation start timestamp if not already present
        headers.setdefault("separation-end", str(t1_model)) # Add separation end timestamp if not already present

        if headers.get("separation-start") and headers.get("separation-end"):
            logger.info(action="inference_timestamp_receiving", status="success", data={"worker_id": worker_id, "filename": file.filename, "separation-start": headers["separation-start"], "separation-end": headers["separation-end"]})
        if not headers.get("separation-start") or not headers.get("separation-end"):
            logger.warning(action="inference_timestamp_receiving", status="failed", data={"worker_id": worker_id, "filename": file.filename, "error": "missing_separation_timestamps"})
        return StreamingResponse(zipstream, media_type="application/zip", headers=headers) #Stream the ZIP file as a response

    except HTTPException:
        raise # Re-raise HTTP exceptions from perform_inference and validate_outputs to preserve status codes

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
    if scnet_model.separator is None:
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