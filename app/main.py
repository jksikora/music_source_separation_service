from fastapi import FastAPI
from app.api.audio_routes import audio_router
from app.api.register_routes import register_router, try_register_request
from contextlib import asynccontextmanager
from app.utils.logging_utils import setup_logging, get_logger
import asyncio

# === Lifespan Event ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup check if SCNet worker config file exists and attempt registration"""
    asyncio.create_task(try_register_request("scnet", 1)) # Attempt to register SCNet worker on startup
    asyncio.create_task(try_register_request("dttnet", 1)) # Attempt to register DTTNet worker on startup
    yield # Pauses here; Code after yield runs on shutdown


app = FastAPI(title = "Music Source Separation Service", lifespan=lifespan)

app.include_router(audio_router) # Include audio processing API routes
app.include_router(register_router) # Include worker registration API routes
setup_logging("DEBUG") # Setup logging configuration
logger = get_logger(__name__)  # Logger for main module


# === Homepage Endpoint ===
@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Homepage for Music Source Separation Service"}