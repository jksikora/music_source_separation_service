from fastapi import FastAPI, Request
from app.api import router as audio_router
from app.services.session_manager import session_manager
from pathlib import Path
import yaml, httpx, asyncio

app = FastAPI(title = "Music Source Separation Service")

app.include_router(audio_router) 

@app.on_event("startup")
async def check_workers():
    asyncio.create_task(try_register_request())

@app.get("/") 
async def root():
    return {"message": "Homepage for Music Source Separation Service"}
    
@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    await session_manager.register_worker(
        worker_id=data["worker_id"],
        model_type=data["model_type"],
        address=data["address"]
    )
    workers = await session_manager.list_workers()
    print(workers)
    return {"status": "registered"}

async def try_register_request():
    config_path = (Path(__file__).resolve().parents[0] / "workers" / "scnet01_config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            worker_config = yaml.safe_load(f)
        worker_address = worker_config.get("address")
    else:
        worker_address = None
        
    if worker_address:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"http://{worker_address}/register_request")
        except Exception:
            print("SCNet worker not reachable — will register later when it starts up.")