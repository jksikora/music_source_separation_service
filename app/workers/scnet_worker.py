from fastapi import FastAPI, UploadFile, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import io
import soundfile as sf
import numpy as np
import zipfile
import importlib.util
import asyncio
from pathlib import Path
from scnet.inference import Seperator, SCNet
from ml_collections import ConfigDict
import yaml
import httpx
from zipstream.ng import ZipStream

app = FastAPI(title = "SCNetWorker")

# --- Load worker config ---

config_path = Path(__file__).resolve().parents[0] / "scnet01_config.yaml"
with open(config_path) as f:
    worker_config = yaml.safe_load(f)

worker_id = worker_config["worker_id"]
model_type = worker_config["model_type"]
main_app_address = worker_config["main_app_address"]
address = worker_config["address"]

# --- Model initialization ---

separator: Seperator | None = None
inference_lock = asyncio.Lock()

@app.on_event("startup")
async def load_model():
    global separator
    
    spec = importlib.util.find_spec("scnet")
    if spec is None:
        raise ImportError("SCNet package not found")
    
    scnet_root = Path(spec.submodule_search_locations[0]).resolve().parent
    config_path = str(scnet_root / "conf" / "config.yaml")

    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = str(repo_root / "checkpoints" / "scnet" / "checkpoint.th")

    with open(config_path, "r") as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        
    model = SCNet(**config.model)
    model.eval()
    separator = Seperator(model, checkpoint_path)

    print(f"[{worker_id}] Model loaded successfully, registering with main app...")

    await try_register()

@app.post(f"/{worker_id}/inference")
async def infer(file: UploadFile):
    try:
        audio = await file.read()
        audio_buffer = io.BytesIO(audio)
        waveform, sample_rate = sf.read(audio_buffer, dtype="float32")
          
        async with inference_lock:
            output_waveforms, output_sample_rates = await asyncio.to_thread(
                separator.separate_music_file, waveform, sample_rate
            )

        chunk_size = 256 * 1024  # 256 KB
        def file_generator(name: str, arr: np.ndarray, sample_rate: int):
            buffer = io.BytesIO()
            sf.write(buffer, arr, sample_rate, format="WAV")
            buffer.seek(0)
            while True:
                chunk = buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
          
        zipstream = ZipStream(sized=True)

        for name, arr in output_waveforms.items():
            zipstream.add(
                file_generator(name, arr, output_sample_rates[name]),
                f"{name}.wav"
            )

        headers = {"Content-Disposition": 'attachment; filename="separated_stems.zip"'}

        return StreamingResponse(zipstream, media_type="application/zip", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register_request")
async def register_request():
    await try_register()
    return {"status": "registered"}

async def try_register():
    if separator is None:
        print("Model not initialized — skipping registration.")
        return

    worker_data = {
        "worker_id": worker_id,
        "model_type": model_type,
        "address": address
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"http://{main_app_address}/register_worker", json=worker_data)
            print("Successfully registered with main service.")
    except Exception:
        print("Main app not running yet — will wait for it to call /register_request later")