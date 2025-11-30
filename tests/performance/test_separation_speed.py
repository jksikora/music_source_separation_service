import numpy as np
import soundfile as sf
import yaml, httpx, os, csv, io, math, random, time

# --- Separation Speed Test ---

# === Load Database ===
db_path = "/mnt/d/studia/praca_inzynierska/database/"


# === Main Address Loaded from YAML File ===
with open("app/workers/scnet/scnet1_config.yaml", "r") as f:
    config = yaml.safe_load(f)
main_address = config["main_address"]  # e.g., "127.0.0.1:8000"


# === Create HTTP Client ===
client = httpx.Client(base_url=f"http://{main_address}", timeout=600.0) # Increased timeout to 600 seconds for long-running model inference


# === Prepare Results Directory and CSV File for Separation Speed Test Results ===
results_dir = "tests/performance/results"
os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist; exist_ok=True avoids error if it already exists
csv_speed = os.path.join(results_dir, "separation_speed_results.csv") # Complete path to separation speed results CSV file

if not os.path.exists(csv_speed):
    with open(csv_speed, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Filename", "Run idx", "Fragment [s]", "Time [s]"])


# === Prepare Model with Warmup Function ===
def warmup_separation(client: httpx.Client) -> None:
    sample_rate = 44100
    duration = 1.0
    amplitude = 0.5
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False) # Time vector; endpoint=False to avoid an extra sample
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32) # Sine wave generation; convert to float32 
    buffer = io.BytesIO()
    sf.write(buffer, sine_wave[:, np.newaxis], sample_rate, format='WAV') # np.newaxis to make it 2D (samples, channels)
    buffer.seek(0)
    audio_bytes = buffer.read()
    files = {'file': ('warmup.wav', audio_bytes, 'audio/wav')}
    response = client.post("/upload_audio", files=files)
    if response.status_code != 200:
        print(f"warmup request failed with status code: {response.status_code}")
    print(f"warmup request completed with status code: {response.status_code}")


# === Collect Audio Files Paths from Database Function ===
def _collect_files(db_path: str) -> list[str]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path does not exist: {db_path}")
    files = [os.path.join(db_path, f) for f in os.listdir(db_path) if f.lower().endswith(".flac")] # Collect all .flac files from database path and return full paths
    files.sort() # Sort files by filename for consistent order
    return files


# === Load Audio Fragments Function ===
def _load_audio_fragments(filename: str, file_path: str) -> list[int]:
    waveform, sample_rate = sf.read(file_path, dtype="float32")
    samples_num = waveform.shape[0]
    duration = samples_num / sample_rate
    max_length_sec = int(math.floor(duration / 10.0) * 10) # Divide duration into segments of 10 seconds, round down to get max length divisible by 10 and multiply by 10 to get max length in seconds
    if max_length_sec < 10:
        raise ValueError(f"Audio file {filename} is shorter than 10 seconds.")

    fragments = list(range(10, max_length_sec + 1, 10)) # Create list of fragment lengths: 10,20,... up to max_length_sec (exclusive so +1)
    print(f"loaded file: {filename}, duration: {duration:.2f}s, fragments: {fragments}")

    return fragments, waveform, sample_rate, samples_num


# === Convert Waveform to Bytes Function ===
def _convert_to_bytes(waveform, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


# === Test Separation Speed Function ===
def test_separation_speed(client: httpx.Client) -> list[tuple[str, dict[int, list[float]]]]:
    results_per_song = []
    files = _collect_files(db_path)
    if not files:
        raise FileNotFoundError(f"No audio files found in database path: {db_path}")
    pass

    for file_path in files:
        filename = os.path.basename(file_path)
        fragments, waveform, sample_rate, samples_num = _load_audio_fragments(filename, file_path)

        timings = {f: [] for f in fragments} # -> dict[fragment_length: list of timings]; initialize empty list for each fragment length
        rng = random.Random(filename)  # Seed random number generator with filename for reproducibility
        for fragment in fragments:
            fragment_samples = int(fragment * sample_rate)
            for i in range(15): # 15 measurements per fragment length
                max_start_idx = samples_num - fragment_samples # Maximum possible start index for fragment
                if max_start_idx <= 0:
                    start = 0
                else:
                    start = rng.randint(0, max_start_idx)

                clip = waveform[start:start + fragment_samples]
                audio_bytes = _convert_to_bytes(clip, sample_rate)
                files = {'file': (filename, audio_bytes, 'audio/wav')}

                t0 = time.time()
                upload_response = client.post("/upload_audio", files=files)
                t1 = time.time()
                if upload_response.status_code != 200:
                    raise RuntimeError(f"unexpected status code from upload_audio endpoint response: {upload_response.status_code} for file: {filename}, fragment: {fragment}s")
                print(f"file: {filename}, fragment: {fragment}s, run: {i+1}, time: {t1 - t0:.4f}s")

                delta_t = t1 - t0
                timings[fragment].append(delta_t)
                time.sleep(0.5) # Short sleep to avoid overwhelming the server

        results_per_song.append((filename, timings))

    return results_per_song
    

# === Save Results to CSV Function ===
def results_to_csv(results_per_song: list[tuple[str, dict[int, list[float]]]], csv_speed: str) -> None:
    # Write detailed per-run results to CSV (one row per measurement)
    with open(csv_speed, "a", newline="") as cf:
        writer = csv.writer(cf)
        for filename, timings in results_per_song:
            for fragment in sorted(timings.keys()):
                runs = timings[fragment]
                for run_idx, result in enumerate(runs, start=1):
                    writer.writerow([filename, run_idx, fragment, f"{result:.6f}"])
                    print(f"saved to CSV: {filename}, fragment: {fragment}s, run: {run_idx}, time: {result:.6f}s")
    

# === Main Execution Block ===
if __name__ == "__main__":
    warmup_separation(client)
    results_per_song = test_separation_speed(client)
    results_to_csv(results_per_song, csv_speed)
