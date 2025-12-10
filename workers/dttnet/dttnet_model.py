from __future__ import annotations
from fastapi import HTTPException, UploadFile
from src.dp_tdf.dp_tdf_net import DPTDFNet
from src.evaluation.separate import no_overlap_inference, overlap_inference  
from app.utils.logging_utils import get_logger
from typing import Dict
from pathlib import Path
import numpy as np
import soundfile as sf
import torch, yaml, asyncio, importlib.util, os, io, time

logger = get_logger(__name__) # Logger for DTTNet Model


# === DTTNet Model Management Class ===
class DTTNetModel:
	"""Class to manage DTTNet model loading and inference."""
	def __init__(self, worker_id: str) -> None:
		self.worker_id = worker_id
		self.inference_lock = asyncio.Lock()
		
		spec = importlib.util.find_spec("src") # Check if DTTNet package is importable
		if spec is None:
			raise ImportError("DTTNet package not found")
		
		self.dttnet_root = Path(spec.submodule_search_locations[0]).resolve().parent # Get DTTNet package root
		self.model_config_path = str(self.dttnet_root / "configs" / "model") # Path to DTTNet default config
		self.infer_config_path = str(self.dttnet_root / "configs" / "evaluation.yaml") # Path to DTTNet default config
		self.worker_root = Path(__file__).resolve().parents[0] # Get the project root
		self.checkpoint_path = str(self.worker_root / "checkpoints") # Path to DTTNet checkpoints folder

		self.sources = {"bass", "drums", "other", "vocals"}  # Default targets
		self.models: Dict[str, DPTDFNet] = {}
		self.loaded = False
		
		with open(self.infer_config_path, "r") as f: # Load DTTNet config file
			infer_cfg = yaml.safe_load(f)

		self.batch_size = infer_cfg.get("batch_size", 4)
		self.double_chunk = infer_cfg.get("double_chunk", False)
		self.overlap_add = infer_cfg.get("overlap_add", None)

		def _select_device(cfg_device: str | None) -> torch.device: # Helper function to select device
			try:
				device = torch.device(cfg_device)
				index = device.index if device.index is not None else 0
				torch.cuda.get_device_properties(index)
				logger.info(action="device_selection", status="success", data={"requested_device": cfg_device, "selected_device": str(device)})
				return device
			except Exception as e:
				logger.warning(action="device_selection", status="failed", data={"requested_device": cfg_device, "fallback_device": "cpu", "error": str(e)})	
				return torch.device("cpu")

		self.device = _select_device(infer_cfg.get("device"))

    # === Model Loading Function ===
	async def load_model(self) -> None:
		"""Instantiate DTTNet checkpoints defined in the config file."""
		for source in self.sources: # Load model for each source
			model_config_path = os.path.join(self.model_config_path, f"{source}.yaml")
			checkpoint_path = os.path.join(self.checkpoint_path, f"{source}.ckpt")

			with open(model_config_path, "r") as f: # Load DTTNet config file
				model_cfg = yaml.safe_load(f)

			target_path = model_cfg.pop("_target_", "src.dp_tdf.dp_tdf_net.DPTDFNet") # Delete _target_ from config to avoid issues
			model = DPTDFNet(**model_cfg) # Unpack model configuration and create DTTNet model instance
			checkpoint = torch.load(checkpoint_path, map_location=self.device) # Load model checkpoint file and map to selected device
			state_dict = checkpoint.get("state_dict", checkpoint) # Get state_dict (actual weights) from checkpoint
			model.load_state_dict(state_dict, strict=True) # Load weights into model instance ensuring all keys match (strict=True)
			model = model.to(self.device) # Move model to the selected device
			model.eval() # Set model to evaluation mode

			self.models[source] = model # Store model instance in the models dictionary keyed by source name

		logger.info(action="model_loading", status="success",data={"worker_id": self.worker_id})
		self.loaded = True  # Mark models as loaded

    # === Inference Function ===
	async def perform_inference(self, file: UploadFile)-> tuple[dict[str, np.ndarray], dict[str, int], float, float]:
		"""Run inference for each configured stem."""	
		if not self.loaded or not self.models:
			logger.error(action="inference", status="failed", data={"worker_id": self.worker_id, "filename": file.filename, "error": "model_not_loaded"})
			raise HTTPException(status_code=503, detail="Model not loaded")
		audio = await file.read() # Read uploaded audio file
		audio_buffer = io.BytesIO(audio) # Create in-memory buffer for audio data
		waveform, sample_rate = sf.read(audio_buffer, dtype="float32")  # Decode audio using soundfile
		mix = self._prepare_input(waveform)

		def _run_separation(mix: np.ndarray, sample_rate: int): # Additional function to run separation for time measurement
			outputs: Dict[str, np.ndarray] = {} # Store separated waveforms
			sample_rates: Dict[str, int] = {} # Store sample rates for each stem
			t0_model = time.time() 
			
			for source, model in self.models.items(): # Run inference for each source
				if self.double_chunk:
					inf_ck = model.inference_chunk_size
				else:
					inf_ck = model.chunk_size
				if self.overlap_add is None:
					target_wav_hat = no_overlap_inference(model, mix, self.device, self.batch_size, inf_ck)
				else:
					if not os.path.exists(self.overlap_add.tmp_root):
						os.makedirs(self.overlap_add.tmp_root)
					target_wav_hat = overlap_inference(model, mix, self.device, self.batch_size, inf_ck, self.overlap_add.overlap_rate, self.overlap_add.tmp_root, self.overlap_add.samplerate)

				outputs[source] = target_wav_hat 
				sample_rates[source] = sample_rate 

			t1_model = time.time()
			return outputs, sample_rates, t0_model, t1_model

		async with self.inference_lock:
			return await asyncio.to_thread(_run_separation, mix, sample_rate)

	# === Prepare Input Function ===
	def _prepare_input(self, waveform: np.ndarray) -> np.ndarray:
		"""Ensure the mixture tensor is shaped (channels, samples)."""
		if waveform.ndim == 1: # Mono
			mix = np.stack([waveform, waveform], axis=0) # Duplicate to create fake stereo
		elif waveform.ndim == 2: # Multi-channel
			mix = waveform.T if waveform.shape[1] <= waveform.shape[0] else waveform # Model expects (channels, samples)
			if mix.shape[0] == 1: # Single channel
				mix = np.vstack([mix, mix]) # Duplicate to create fake stereo
			elif mix.shape[0] > 2: # More than 2 channels
				mix = mix[:2, :] # Use only first two channels
		else: 
			raise HTTPException(status_code=400, detail="Unsupported audio shape")
		
		return mix
	
    # === Check if Model is Loaded ===
	def is_loaded(self) -> bool:
		"""Check if the DTTNet models are loaded and ready for inference."""
		return self.loaded and bool(self.models)