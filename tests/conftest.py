from app.services.storage import storage
import numpy as np
import soundfile as sf
import pytest, pytest_asyncio, io, httpx, yaml

# === Main Address Loaded from YAML File ===
with open("app/workers/scnet/scnet1_config.yaml", "r") as f:
    config = yaml.safe_load(f)
main_address = config["main_address"]  # e.g., "127.0.0.1:8000"


# === Test Client Fixture ===
@pytest.fixture
def client():
    """For each test create a different HTTP client to interact with the running FastAPI app (synchronously) on the configured address"""
    with httpx.Client(base_url=f"http://{main_address}") as client:
        yield client


# === Async Test Client Fixture ===
@pytest_asyncio.fixture
async def async_client():
    """For each test create a different async HTTP client to interact with the running FastAPI app (asynchronously) on the configured address"""
    async with httpx.AsyncClient(base_url=f"http://{main_address}") as client:
        yield client


# === Invalid Sample Audio File Fixture ===
@pytest.fixture
def sample_audio_file(request):
    """Generate an invalid sample audio file for simulating uploads with invalid data"""
    format = request.param if hasattr(request, 'param') else 'WAV'  # Will default to 'wav' if no param is provided
    sample_rate = 44100
    waveform = np.random.randn(sample_rate).astype(np.float32) * 0.1 # 1 seconds of random audio noise (soundfile and torchaudio expect float32 format)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format=format) # waveform instead of waveform.T.numpy(), because it's already a numpy array
    buffer.seek(0)
    
    return buffer.read(), sample_rate, format


# === Sample Audio File Fixture ===
@pytest.fixture
def invalid_sample_audio_file(request):
    """Generate a sample audio file (1 second of random noise) for simulating uploads"""
    format = request.param if hasattr(request, 'param') else 'txt'  # Will default to 'txt' if no param is provided
    filename = f"test.{format.lower()}"
    data = b"Invalid file."
    
    if format == "txt":
        return filename, data, "text/plain"
    elif format in {"png", "jpg", "svg"}:
        match format:
            case "png":
                return filename, data, "image/png"
            case "jpg":
                return filename, data, "image/jpeg"
            case "svg":
                return filename, data, "image/svg+xml"
    elif format in {"mp4", "avi", "mov"}:
        match format:
            case "mp4":
                return filename, data, "video/mp4"
            case "avi":
                return filename, data, "video/x-msvideo"
            case "mov":
                return filename, data, "video/quicktime"


# === Sample Audio File in Storage Fixture ===
@pytest_asyncio.fixture
async def sample_audio_file_in_storage(sample_audio_file):
    """Save a sample audio file in storage for testing storage service"""
    audio_bytes, sample_rate, _ = sample_audio_file
    file_id = "test_file_id"
    filename = f"test.wav"
    await storage.save(file_id, filename, audio_bytes, sample_rate)
    yield file_id