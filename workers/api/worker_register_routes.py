from fastapi import APIRouter, HTTPException
from app.schemas.worker_schemas import WorkerConfig, Worker
from app.utils.logging_utils import get_logger
import httpx

worker_register_router = APIRouter() # Create API router for worker register routes
logger = get_logger(__name__) # Logger for worker_register_routes module


# === Register Request Endpoint ===
@worker_register_router.post("/register_request")
async def register_request(data: WorkerConfig) -> None:
    """Endpoint for main app to request worker registration in Session Manager"""
    try:
        await try_register(data.worker_id, data.model_type, data.worker_address, data.main_address)
        logger.info(action="registration_request", status="success", data={"worker_id": data.worker_id})

    except Exception:
        logger.warning(action="registration_request", status="failed", data={"worker_id": data.worker_id, "status_code": 500, "error": "registration_failed"})
        raise HTTPException(status_code=500, detail="Registration attempt failed")


# === Try Register Worker Function ===
async def try_register(worker_id: str, model_type: str, worker_address: str, main_address: str) -> None:
    """Function to attempt registering worker with main app in Session Manager if model is initialized"""
    worker_data = Worker(
        worker_id=worker_id,
        model_type=model_type,
        worker_address=worker_address,
    )
    
    try:
        async with httpx.AsyncClient(timeout=10) as client: # HTTP client with timeout
            response = await client.post(f"http://{main_address}/register_worker", json=worker_data.model_dump()) # Send registration request to main app; model_dump() converts Pydantic model to Python dict
        if response.status_code == 200:
            logger.info(action="registration_attempt", status="success", data={"worker_id": worker_id, "model_type": model_type, "address": worker_address})
        else:
            logger.warning(action="registration_attempt", status="failed", data={"worker_id": worker_id, "status_code": response.status_code, "error": response.text})
    
    except Exception as e:
        logger.warning(action="registration_attempt", status="failed", data={"worker_id": worker_id, "status_code": 500, "error": str(e)})