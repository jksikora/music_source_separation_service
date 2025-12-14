from fastapi import APIRouter, HTTPException
from app.schemas.worker_schemas import WorkerConfig, Worker
from app.utils.logging_utils import get_logger
import httpx, asyncio, time

worker_register_router = APIRouter() # Create API router for worker register routes
logger = get_logger(__name__) # Logger for worker_register_routes module


# === Register Request Endpoint ===
@worker_register_router.post("/register_request")
async def register_request(data: WorkerConfig) -> None:
    """Endpoint for main app to request worker registration in Session Manager"""
    worker_data = Worker(
        worker_id=data.worker_id,
        model_type=data.model_type,
        worker_address=data.worker_address,
    )
    
    try:
        async with httpx.AsyncClient(timeout=10) as client: # HTTP client with timeout
            response = await client.post(f"http://{data.main_address}/register_worker", json=worker_data.model_dump()) # Send registration request to main app; model_dump() converts Pydantic model to Python dict
        if response.status_code == 200:
            logger.info(action="registration_request", status="success", data={"worker_id": data.worker_id, "model_type": data.model_type, "address": data.worker_address})
        else:
            logger.warning(action="registration_request", status="failed", data={"worker_id": data.worker_id, "model_type": data.model_type, "address": data.worker_address, "status_code": response.status_code, "error": response.text})
    except httpx.RequestError as e:
        logger.warning(action="registration_request", status="failed", data={"worker_id": data.worker_id, "model_type": data.model_type, "address": data.worker_address, "status_code": 500, "error": str(e)})
        raise HTTPException(status_code=500, detail="Registration request failed")


# === Try Register Worker Function ===
async def try_register(worker_id: str, model_type: str, worker_address: str, main_address: str) -> None:
    """Function to attempt registering worker with main app in Session Manager"""
    worker_data = Worker(
        worker_id=worker_id,
        model_type=model_type,
        worker_address=worker_address,
    )

    backoff = 0.5
    max_backoff = 2.0
    deadline = 60.0
    start_time = time.time()
    last_error = None
    while (time.time() - start_time) < deadline: # Retry loop with deadline for more robust checking if main app is reachable
        try: # Check if the main app's /register_worker endpoint is reachable before sending registration request
            async with httpx.AsyncClient(timeout=10) as client: # HTTP client with timeout
                response = await client.post(f"http://{main_address}/register_worker", json=worker_data.model_dump()) # Send registration request to main app; model_dump() converts Pydantic model to Python dict
            if response.status_code == 200:
                logger.info(action="registration_attempt", status="success", data={"worker_id": worker_id, "model_type": model_type, "address": worker_address})
                return # Exit loop on successful registration
            else:
                last_error = response.text # Set error message for inner else case (for e.g., service is reachable but registration failed)
                logger.warning(action="registration_attempt", status="in progress", data={"worker_id": worker_id, "status_code": response.status_code, "details": response.text})
        except httpx.RequestError as e:
            last_error = str(e) # Set error message for network/timeout errors
            logger.warning(action="registration_attempt", status="in progress", data={"worker_id": worker_id, "main_address": main_address, "status_code": 500, "error": str(e)})
        await asyncio.sleep(backoff) # Wait before retrying next registration attempt
        backoff = min(backoff * 1.5, max_backoff)  # Exponential backoff until max 2 seconds
    else:
        logger.warning(action="registration_attempt", status="failed", data={"worker_id": worker_id, "error": "deadline_exceeded", "details": last_error})