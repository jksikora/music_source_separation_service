from typing import Generator
from fastapi.testclient import TestClient
from app.main import app
from app.services.storage import storage
from app.services.session_manager import session_manager
import numpy as np
import soundfile as sf
import pytest, io, httpx, yaml

# Load config to get main address
with open("app/workers/scnet/scnet1_config.yaml", "r") as f:
    config = yaml.safe_load(f)
main_address = config["main_address"]  # e.g., "127.0.0.1:8000"

# === Test Client Fixture ===
@pytest.fixture
def client():
    """For each test create a different HTTP client to interact with the running FastAPI app on the configured address"""
    with httpx.Client(base_url=f"http://{main_address}") as client:
        yield client


# === Sample Audio File Fixture ===
@pytest.fixture
def sample_audio_file(request) -> tuple[bytes, int]:
    """Generate a sample audio file (1 second of random noise) for simulating uploads"""
    format = request.param if hasattr(request, 'param') else 'WAV'  # Will default to 'wav' if no param is provided
    fs=44100
    waveform = np.random.randn(fs).astype(np.float32) * 0.1 # 1 seconds of random audio noise (soundfile and torchaudio expect float32 format)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, fs, format=format) # waveform instead of waveform.T.numpy(), because it's already a numpy array
    buffer.seek(0)
    
    return buffer.read(), fs, format