from fastapi import HTTPException
from app.utils.logging_utils import get_logger
from zipstream.ng import ZipStream
import numpy as np
import soundfile as sf
from typing import Generator
import io

logger = get_logger(__name__) # Logger for main module


# === Validate Inference Outputs Function ===
def validate_outputs(output_waveforms: dict[str, np.ndarray], output_sample_rates: dict[str, int], worker_id: str, filename: str) -> None:
    """Function to validate that inference outputs contain non-empty waveforms and matching sample rates"""
    if not output_waveforms: # Check if any stems were returned
        logger.error(action="inference_validation", status="failed", data={"worker_id": worker_id, "filename": filename, "error": "no_stems_returned"})
        raise HTTPException(status_code=500, detail="Inference produced no stems")

    mismatched_stems = set(output_waveforms) ^ set(output_sample_rates) # Check if all stems have corresponding sample rates
    if mismatched_stems:
        logger.error(action="inference_validation", status="failed", data={"worker_id": worker_id, "filename": filename, "mismatched_stems": mismatched_stems, "error": "waveform_sample_rate_mismatch"})
        raise HTTPException(status_code=500, detail="Inference results inconsistent")

    invalid_waveforms = [name for name, waveform in output_waveforms.items() if waveform is None or not isinstance(waveform, np.ndarray) or waveform.size == 0] # Check for invalid waveforms
    if invalid_waveforms:
        logger.error(action="inference_validation", status="failed", data={"worker_id": worker_id, "filename": filename, "invalid_stems": invalid_waveforms, "error": "invalid_waveforms_returned"})
        raise HTTPException(status_code=500, detail="Inference returned invalid waveforms")

    invalid_sample_rates = [name for name, sample_rate in output_sample_rates.items() if sample_rate is None or sample_rate <= 0] # Check for invalid sample rates
    if invalid_sample_rates:
        logger.error(action="inference_validation", status="failed", data={"worker_id": worker_id, "filename": filename, "invalid_stems": invalid_sample_rates, "error": "invalid_sample_rates_returned"})
        raise HTTPException(status_code=500, detail="Inference returned invalid sample rates")

    stem_names = ', '.join(output_waveforms.keys()) # Prepare stem names for logging
    logger.info(action="inference_validation", status="success", data={"worker_id": worker_id, "filename": filename, "stems": stem_names})


# === ZIP Stream Generator Function ===
def zipstream_generator(waveforms: dict[str, np.ndarray], sample_rates: dict[str, int], worker_id: str, filename: str) -> tuple[ZipStream, dict[str, str]]:
    """Function to create a streaming ZIP plus headers for separated stems with consistent logging"""
    zipstream = ZipStream(sized=True)
    for name, waveform in waveforms.items():
        zipstream.add(_buffer_generator(waveform, sample_rates[name]), f"{name}.wav")
        logger.debug(action="stem_addition", status="success", data={"worker_id": worker_id, "stem": name})

    headers = {"Content-Disposition": f'attachment; filename="{filename}_separated_stems.zip"'}
    return zipstream, headers


# === Helper Buffer Generator Function ===
def _buffer_generator(waveform: np.ndarray, sample_rate: int, chunk_size: int = 256 * 1024) -> Generator[bytes, None, None]:
    """Helper function to yield WAV-encoded chunks without loading entire response in memory"""
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    buffer.seek(0)
    while True:
        chunk = buffer.read(chunk_size)
        if not chunk:
            break
        yield chunk