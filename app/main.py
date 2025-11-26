from fastapi import FastAPI, HTTPException
from app.api.audio_routes import router as audio_router
from app.services.session_manager import session_manager
from app.schemas.worker_schemas import Worker
from app.utils.config_utils import load_worker_config
from app.utils.logging_utils import setup_logging, get_logger
import asyncio, httpx

app = FastAPI(title = "Music Source Separation Service")

app.include_router(audio_router) # Include audio processing API routes
setup_logging() # Setup logging configuration
logger = get_logger(__name__)  # Logger for main module


# === Startup Event ===
@app.on_event("startup")
async def check_workers() -> None:
    """On startup check if SCNet worker config file exists and attempt registration on startup"""
    asyncio.create_task(try_register_request("scnet", 1))


# === Homepage Endpoint ===
@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Homepage for Music Source Separation Service"}


# === Register Worker Endpoint ===
@app.post("/register_worker")
async def register_worker(worker_data: Worker) -> None:
    """Endpoint for workers to register themselves in Session Manager"""
    try:
        await session_manager.register_worker(
            worker_id=worker_data.worker_id,
            model_type=worker_data.model_type,
            worker_address=worker_data.worker_address
        )
        logger.info(action="worker_registration", status="success", data={"worker_id": worker_data.worker_id, "model_type": worker_data.model_type, "worker_address": worker_data.worker_address})
    except Exception as exc:
        logger.exception(action="worker_registration", status="failed", data={"worker_id": worker_data.worker_id, "status_code": 500, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Worker registration failed")


# === Try Register SCNet Worker Function ===
async def try_register_request(model: str, serial_number: int) -> None:
    """Function to attempt registering SCNet worker in Session Manager if config file exists"""
    try:
        worker_config = load_worker_config(model, serial_number)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(action="registration_request", status="failed", data={"error": "invalid_worker_config", "details": str(exc)})
        return

    try:
        async with httpx.AsyncClient(timeout=10) as client: # create HTTP client with timeout
            response = await client.post(f"http://{worker_config.worker_address}/register_request") # Send registration request to SCNet worker
        if response.status_code == 200:
            logger.info(action="registration_request", status="success", data={"address": worker_config.worker_address})
        else:
            logger.warning(action="registration_request", status="failed", data={"address": worker_config.worker_address, "status_code": response.status_code, "error": response.text})

    except Exception as exc:
        logger.warning(action="registration_request", status="failed", data={"address": worker_config.worker_address, "status_code": 500, "error": str(exc)})