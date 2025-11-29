import soundfile as sf
import numpy as np
import musdb, museval, io, csv, os, yaml, httpx, json
import matplotlib.pyplot as plt

# --- Separation Quality Benchmark Test ---

# === Load MUSDB18-HQ Dataset ===
mus = musdb.DB(root="/mnt/d/studia/praca_inzynierska/musdb18hq", is_wav=True, subsets="test")


# === Main Address Loaded from YAML File ===
with open("app/workers/scnet/scnet1_config.yaml", "r") as f:
    config = yaml.safe_load(f)
main_address = config["main_address"]  # e.g., "127.0.0.1:8000"


# === Create HTTP Client ===
client = httpx.Client(base_url=f"http://{main_address}", timeout=600.0) # Increased timeout to 600 seconds for long-running model inference


# === Prepare Results Directory and CSV File for Separation Quality Test Results ===
results_dir = "tests/performance/results"
os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist; exist_ok=True avoids error if it already exists
csv_file = os.path.join(results_dir, "separation_quality_results.csv") # Complete path to CSV file

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Stem", "Median SDR", "Median SIR", "Median SAR", "Median ISR"])


# === Convert Audio to Bytes Helper Function ===
def audio_to_bytes(audio, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


# === Test Separation Quality Function ===
def test_separation_quality():
    for track in mus.tracks[:2]:  # Limit to first 1 for testing; remove [:1] for full test
        print(f"\nstart processing file: {track.name}")
        
        input_audio_bytes = audio_to_bytes(track.audio, track.rate) # Convert musdb mixture from numpy.ndarray to bytes
        files = {'file': (f'{track.name}.wav', input_audio_bytes, 'audio/wav')}
        upload_response = client.post("/upload_audio", files=files)

        if upload_response.status_code != 200:
            print(f"unexpected status code from upload_audio endpoint response: {upload_response.status_code} for file: {track.name}")
            continue # Skip to next track on failure
        
        data = upload_response.json()
        estimates = {}
        for stem_name, stem_data in data.items():
            file_id = stem_data['file_id']
            download_resp = client.get(f"/download_audio/{file_id}")
            if download_resp.status_code == 200:
                buffer = io.BytesIO(download_resp.content)
                output_waveform, _ = sf.read(buffer)
                estimates[stem_name] = output_waveform # Transpose because museval expects (channels, samples)
                print(f"estimated stem {stem_name} of file: {track.name} shape: {estimates[stem_name].shape}")
                print(f"reference stem {stem_name} of file: {track.name} shape: {track.targets[stem_name].audio.shape}")

            else:
                print(f"unexpected status code from download_audio endpoint response: {download_resp.status_code} for {stem_name} of file: {track.name}")
            
        try:
            museval.eval_mus_track(track, estimates, output_dir=results_dir)

            test_output = os.path.join(results_dir, "test", f"{track.name}.json")
            metrics = ["SDR", "SIR", "SAR", "ISR"]
            with open(test_output, 'r') as jf:
                data = json.load(jf)
                
            with open(csv_file, 'a', newline='') as cf:
                writer = csv.writer(cf)
                for target in data.get("targets"):
                    stem_name = target.get("name")
                    frames = target.get("frames")
                    metrics_medians = []
                    for metric in metrics:
                        values = [frame["metrics"][metric] for frame in frames if np.isfinite(frame["metrics"][metric])] # Filter out NaN values
                        metrics_medians.append(np.median(values))
                    writer.writerow([track.name, stem_name] + metrics_medians)
                    print(f"Medians for {stem_name}: SDR={metrics_medians[0]:.2f}, SIR={metrics_medians[1]:.2f}, SAR={metrics_medians[2]:.2f}, ISR={metrics_medians[3]:.2f}")

        except Exception as e:
            print(f"evaluation failed for: {track.name} with error: {e}")


if __name__ == "__main__":
    test_separation_quality()