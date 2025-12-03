from fastapi import UploadFile, HTTPException
from scnet.inference import Seperator, SCNet
from app.utils.logging_utils import get_logger
from ml_collections import ConfigDict
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml, asyncio, importlib.util, io, time

logger = get_logger(__name__) # Logger for SCNet Model


# === SCNet Model Management Class ===
class SCNetModel:
    """Class to manage SCNet model loading and inference."""
    def __init__(self):
        self.separator: Seperator | None = None
        self.inference_lock = asyncio.Lock()
    
    # === Model Loading Function ===
    async def load_model(self, worker_id: str, model_type: str) -> None:
        """Function to load SCNet model, initialize separator instance on startup and try registering worker"""
        spec = importlib.util.find_spec("scnet") # Check if SCNet package is importable
        if spec is None:
            raise ImportError("SCNet package not found")
        
        scnet_root = Path(spec.submodule_search_locations[0]).resolve().parent # Get SCNet package root
        config_path = str(scnet_root / "conf" / "config.yaml") # Path to SCNet default config
        worker_root = Path(__file__).resolve().parents[0] # Get the project root
        checkpoint_path = str(worker_root / "checkpoints" / "checkpoint.th") # Path to SCNet checkpoint

        with open(config_path, "r") as f: # Load SCNet config file
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader)) # Load YAML content
            
        model = SCNet(**config.model) # Unpack model configuration and create SCNet model instance
        model.eval() # Set model to evaluation mode
        self.separator = Seperator(model, checkpoint_path) # Load model checkpoint into Seperator instance, select device automatically (CPU/GPU) and prepare for inference

        logger.info(action="model_loading", status="success", data={"worker_id": worker_id, "model_type": model_type})

    # === Inference Function ===
    async def perform_inference(self, file: UploadFile, worker_id: str) -> tuple[dict[str, np.ndarray], dict[str, int], float, float]:
        """Perform inference and return results."""
        if self.separator is None: # Check if separator is initialized
            logger.error(action="inference", status="failed", data={"worker_id": worker_id, "filename": file.filename, "error": "model_not_loaded"})
            raise HTTPException(status_code=503, detail="Model not loaded")

        audio = await file.read() # Read uploaded audio file
        audio_buffer = io.BytesIO(audio) # Create in-memory buffer for audio data
        waveform, sample_rate = sf.read(audio_buffer, dtype="float32") # Read audio data from buffer
        
        def _run_separation(waveform_local, sample_rate_local): # Additional function to run separation for time measurement
            t0_model = time.time()
            outputs = self.separator.separate_music_file(waveform_local, sample_rate_local)
            t1_model = time.time()
            return (*outputs, t0_model, t1_model)

        async with self.inference_lock: # Acquire lock to serialize inference requests
            return await asyncio.to_thread(_run_separation, waveform, sample_rate) # Perform inference in a separate thread and return timestamps
    
    # === Check if Model is Loaded ===
    def is_loaded(self) -> bool:
        """Check if the SCNet model is loaded and ready for inference."""
        return self.separator is not None