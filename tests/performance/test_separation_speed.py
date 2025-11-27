# performance means non-functional tests: latency, throughput, memory; here speed of separation is tested

# === speed of separation test ===
# warmup run 
# load 15 records 
# set time point for start; 10 seconds
# run separation 30 times
# set next time point; 20 seconds
# run separation 30 times
# etc.
# save results (filename, file size, full file time, time point, each one separation time) to csv file 

# === separation quality test ===
import time
import pytest

@pytest.mark.parametrize("sample_audio_file", ["WAV", "FLAC"], indirect=True)
def test_separation_speed(client, sample_audio_file):
    """Test that audio separation completes within a reasonable time (e.g., < 10 seconds for 1-second audio)."""
    audio_bytes, _, format = sample_audio_file
    files = {'file': (f'test.{format.lower()}', audio_bytes, f'audio/{format.lower()}')}
    
    start_time = time.time()
    response = client.post("/upload_audio", files=files)
    end_time = time.time()
    
    separation_time = end_time - start_time
    print(f"Separation time for {format}: {separation_time:.2f} seconds")
    
    # Assert reasonable speed (adjust threshold based on your hardware/model)
    assert separation_time < 10.0, f"Separation took too long: {separation_time:.2f} seconds"
    assert response.status_code == 200