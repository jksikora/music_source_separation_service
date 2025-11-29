import soundfile as sf
import numpy as np
import musdb, museval, io, csv, os, yaml, httpx

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
        writer.writerow(["filename", "stem", "SDR"])


# === Convert Audio to Bytes Helper Function ===
def audio_to_bytes(audio, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


# === Test Separation Quality Function ===
def test_separation_quality():
    for track in mus.tracks[:1]:  # Limit to first 1 for testing; remove [:1] for full test
        print(f"\nstart processing file: {track.name}")
        
        input_waveform = track.audio # Get mixture audio from musdb track
        input_sample_rate = track.rate
        input_audio_bytes = audio_to_bytes(input_waveform, input_sample_rate) # Convert mixture from numpy.ndarray to bytes
        
        files = {'file': (f'{track.name}.wav', input_audio_bytes, 'audio/wav')}
        upload_response = client.post("/upload_audio", files=files)
        if upload_response.status_code != 200:
            print(f"unexpected status code from upload_audio endpoint response: {upload_response.status_code} for file: {track.name}")
            continue # Skip to next track on failure
        
        data = upload_response.json()
        print("SERVER RETURN ORDER:", list(data.keys()))

        estimates = {}
        references = {}
        stem_order = list(data.keys())  # Use the order returned by the server
        for stem_name in stem_order:
            file_id = data[stem_name]['file_id']
            download_resp = client.get(f"/download_audio/{file_id}")
            if download_resp.status_code == 200:
                buffer = io.BytesIO(download_resp.content)
                output_waveform, _ = sf.read(buffer)
                estimates[stem_name] = output_waveform # Transpose because museval expects (channels, samples)
                print(f"estimated stem {stem_name} of file: {track.name} shape: {estimates[stem_name].shape}")
            else:
                print(f"unexpected status code from download_audio endpoint response: {download_resp.status_code} for {stem_name} of file: {track.name}")
            
            references[stem_name] = track.targets[stem_name].audio  # Transpose because museval expects (channels, samples)
            print(f"reference stem {stem_name} of file: {track.name} shape: {track.targets[stem_name].audio.shape}")
        
        try:
            ref_arr = np.stack([references[s] for s in stem_order])
            est_arr = np.stack([estimates[s] for s in stem_order])
            print(f"ref_arr shape: {ref_arr.shape}, est_arr shape: {est_arr.shape}")

            print("museval.evaluate =", museval.evaluate)

            results = museval.evaluate(ref_arr, est_arr) # Results is a list of DataFrames, one DataFrame per stem; Default frame length and hop equals 1 second
            print(type(results[0]))
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for stem_name, dataframe in zip(stem_order, results):
                    sdr = dataframe["SDR"].median()  # According to The 2018 Signal Separation Evaluation Campaign paper, median is preferred over mean for SDR
                    writer.writerow([track.name, stem_name, sdr])
                    assert sdr is not None, f"SDR is None for stem {stem_name} of file: {track.name}"
                    print(f" SDR = {sdr:.2f} for stem {stem_name} of file: {track.name}")
                # for stem_name in stem_order:
                #     print(stem_name, "silent:", np.allclose(references[stem_name], 0))
                #     sdr_median = results[stem_order.index(stem_name)]['SDR'].median()  # According to The 2018 Signal Separation Evaluation Campaign paper, median is preferred over mean for SDR
                #     print(f"reference stem {stem_name} shape:{references[stem_name].shape}", f"estimate stem {stem_name} shape:{estimates[stem_name].shape}")
                #     writer.writerow([track.name, stem_name, sdr_median])
                #     assert sdr_median is not None, f"SDR is None for stem {stem_name} of file: {track.name}"
                #     print(f" SDR = {sdr_median:.2f} for stem {stem_name} of file: {track.name}")

        except Exception as e:
            assert False, f"evaluation failed for: {track.name}; {e}"


if __name__ == "__main__":
    test_separation_quality()
