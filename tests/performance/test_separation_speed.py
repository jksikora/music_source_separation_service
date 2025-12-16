import soundfile as sf
import numpy as np
from pathlib import Path
import gdown
import yaml, httpx, os, csv, io, math, random, time, requests, zipfile, argparse

# --- Separation Speed Test ---

# === Database Loader Function ===
def get_database(root_path=None):
    if root_path is None:
        tests_dir = Path(__file__).resolve().parents[1]  # Get the 'tests' directory
        root_path = tests_dir / "database" # Default path to store database: 'tests/database'
    else:
        root_path = Path(root_path) # Convert provided string to Path object
    root_path.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist; parents=True to create any necessary parent directories, exist_ok=True avoids error if it already exists

    if not any(root_path.iterdir()): # Check if directory is empty
        zip_path = root_path / "database.zip" # Create a path for the downloaded zip file
        url = "https://drive.google.com/uc?id=1NJrrlGa2HhB1VhbfOYZMLfSVXiPcanU8"

        print(f"Downloading database to {zip_path} ...")
        gdown.download(url, str(zip_path), quiet=False) # Download the zip file from Google Drive; quiet=False to show progress bar in console

        print(f"Unpacking 'database.zip' to {root_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as z: # Open the downloaded zip file for reading
            z.extractall(root_path) # Extract all contents to the root_path
        os.remove(zip_path) # Remove the zip file after extraction
        
        print(f"Database successfully downloaded and unpacked to {root_path}")

    return str(root_path)

# === Main Address Loaded from YAML File ===
with open("workers/scnet/scnet1_config.yaml", "r") as f:
    config = yaml.safe_load(f)
main_address = config["main_address"]  # e.g., "127.0.0.1:8000"; If tests are run on the host (not inside Docker) the compose service name `main:8000` is not resolvable from the host

if isinstance(main_address, str) and main_address.startswith("main:"): # When the config contains the compose service name, 
    _, port = main_address.split(":", 1) # translate it to the host address so tests can connect to the published port (localhost)
    main_address = f"127.0.0.1:{port}"


# === Create HTTP Client ===
client = httpx.Client(base_url=f"http://{main_address}", timeout=600.0) # Increased timeout to 600 seconds for long-running model inference


# === Prepare Results Directory and CSV Files for Separation Speed Test Results ===
results_dir = "tests/performance/separation_speed"
os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist; exist_ok=True avoids error if it already exists

def get_result_files(prefix: str):
    csv_client = os.path.join(results_dir, f"{prefix}_client_separation_speed_results.csv") # Complete path to client separation speed results CSV file
    csv_model = os.path.join(results_dir, f"{prefix}_model_separation_speed_results.csv") # Complete path to model separation speed results CSV file
    agg_csv_client = os.path.join(results_dir, f"{prefix}_agg_client_separation_speed_results.csv") # Complete path to aggregated client separation speed results CSV file
    agg_csv_model = os.path.join(results_dir, f"{prefix}_agg_model_separation_speed_results.csv") # Complete path to aggregated model separation speed results CSV file

    if not os.path.exists(csv_client):
        with open(csv_client, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["Filename", "Run idx", "Fragment [s]", "Time [s]"])

    if not os.path.exists(csv_model):
        with open(csv_model, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["Filename", "Run idx", "Fragment [s]", "Time [s]"])
    
    return csv_client, csv_model, agg_csv_client, agg_csv_model


# === Prepare Model with Warmup Function ===
def warmup_separation() -> None:
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
    for model in ["scnet", "dttnet"]:
        response = client.post(f"/upload_audio/{model}", files=files)
        if response.status_code != 200:
            print(f"warmup request failed with status code: {response.status_code}")
        else:
            print(f"warmup request completed with status code: {response.status_code}")


# === Collect Audio Files Paths from Database Function ===
def _collect_files(db_path: str) -> list[str]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database path does not exist: {db_path}")
    files = [os.path.join(db_path, f) for f in os.listdir(db_path) if f.lower().endswith(".flac")] # Collect all .flac files from database path and return full paths
    files.sort() # Sort files by filename for consistent order
    return files # Limit to first n songs by adding [:n]


# === Load Audio Fragments Function ===
def _load_audio_fragments(filename: str, file_path: str) -> list[int]:
    waveform, sample_rate = sf.read(file_path, dtype="float32")
    samples_num = waveform.shape[0]
    duration = samples_num / sample_rate
    max_length_sec = int(math.floor(duration / 30.0) * 30) # Divide duration into segments of 30 seconds, round down to get max length divisible by 30 and multiply by 30 to get max length in seconds
    if max_length_sec < 30:
        raise ValueError(f"audio file {filename} is shorter than 30 seconds.")

    fragments = list(range(30, max_length_sec + 1, 30)) # Create list of fragment lengths: 30,60,... up to max_length_sec (exclusive so +1)
    print(f"loaded file: {filename}, duration: {duration:.2f}s, fragments: {fragments}")

    return fragments, waveform, sample_rate, samples_num


# === Convert Waveform to Bytes Function ===
def _convert_to_bytes(waveform, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


# === Test Separation Speed Function ===
def test_separation_speed(model: str, prefix: str) -> list[tuple[str, dict[int, list[float]]]]:
    results_per_song_client = []  
    results_per_song_model = []  
    files = _collect_files(db_path)
    if not files:
        raise FileNotFoundError(f"no audio files found in database path: {db_path}")

    for file_path in files:
        filename = Path(file_path).stem
        fragments, waveform, sample_rate, samples_num = _load_audio_fragments(filename, file_path)

        timings_client = {f: [] for f in fragments} # dict[fragment_length, list[client times]]; initialize empty lists for each fragment length
        timings_model = {f: [] for f in fragments}  # dict[fragment_length, list[model times]]; initialize empty lists for each fragment length
        rng = random.Random(filename)  # Seed random number generator with filename for reproducibility
        for fragment in fragments:
            fragment_samples = int(fragment * sample_rate)
            for i in range(5): # Set number of runs, now: 5 measurements per fragment length
                max_start_idx = samples_num - fragment_samples # Maximum possible start index for fragment
                if max_start_idx <= 0:
                    start = 0
                else:
                    start = rng.randint(0, max_start_idx)

                clip = waveform[start:start + fragment_samples]
                audio_bytes = _convert_to_bytes(clip, sample_rate)
                files = {'file': (filename, audio_bytes, 'audio/wav')}

                try: # Added try-except to allow saving partial results
                    t0_client = time.time()
                    upload_response = client.post(f"/upload_audio/{model}", files=files)
                    t1_client = time.time()
                    if upload_response.status_code != 200:
                        raise RuntimeError(f"unexpected status code from upload_audio endpoint response: {upload_response.status_code} for file: {filename}, fragment: {fragment}s")

                    delta_t_client = t1_client - t0_client
                    timings_client[fragment].append(delta_t_client)
                    print(f"file: {filename}, fragment: {fragment}s, run: {i+1}, delta_t_client: {delta_t_client:.4f}s")

                    t0_model = upload_response.headers.get("Separation-Start")
                    t1_model = upload_response.headers.get("Separation-End")
                    if t0_model is not None and t1_model is not None:
                        try:
                            delta_t_model = float(t1_model) - float(t0_model)
                            timings_model[fragment].append(delta_t_model)
                            print(f"file: {filename}, fragment: {fragment}s, run: {i+1}, delta_t_model: {delta_t_model:.4f}s")
                        except Exception as e:
                            raise RuntimeError(f"inference timestamp parsing failed for file: {filename}, fragment: {fragment}s, run: {i+1}, with error: {e}")
                    else:
                        raise RuntimeError(f"inference timestamps not provided for file: {filename}, fragment: {fragment}s, run: {i+1}")
                    time.sleep(0.5) # Short sleep to avoid overwhelming the server

                except (httpx.RequestError, RuntimeError, Exception, KeyboardInterrupt) as e: # If the server stops or any error occurs during measurement, save collected results
                    print(f"measurement interrupted for file {filename}, fragment {fragment}s, run {i+1}: {e}")

                    if results_per_song_client or results_per_song_model:
                        print("saving results after interruption")
                        return results_per_song_client, results_per_song_model
                    
                    else:
                        print("no results to save after interruption")
                        break
                
        results_per_song_client.append((filename, timings_client))
        results_per_song_model.append((filename, timings_model))

    return results_per_song_client, results_per_song_model
    

# === Save Results to CSV Function ===
def results_to_csv(csv_file: str, results_per_song: list[tuple[str, dict[int, list[float]]]]) -> None:
    with open(csv_file, "a", newline="") as cf:
        writer = csv.writer(cf)
        for filename, timings in results_per_song:
            for fragment in sorted(timings.keys()):
                runs = timings[fragment]
                for run_idx, result in enumerate(runs, start=1):
                    writer.writerow([filename, run_idx, fragment, f"{result:.6f}"])
                    print(f"saved to CSV: {filename}, fragment: {fragment}s, run: {run_idx}, time: {result:.6f}s")
    

# === Aggregate Results to CSV Function ===
def aggregate_results_to_csv(csv_input: str, csv_output: str) -> None:
    agg = {}  # dict[(filename, fragment_length), list of times]
    if not os.path.exists(csv_input):
        print(f"csv file does not exist: {csv_input}")
        return

    with open(csv_input, "r", newline="") as incf:
        reader = csv.reader(incf)
        header = next(reader, None) # Skip header row so we only process data rows
        for row in reader:
            try:
                filename = row[0] 
                fragment = float(row[2])
                time_str = row[3]
                t = float(time_str)
                key = (filename, int(fragment)) # Building a grouping key based on filename and fragment length with fragment as int for consistency
                agg.setdefault(key, []).append(t) # Check if key exists, if not create an empty list, then append time to the list for this (filename, fragment) key
            except Exception as e:
                print(f"failed to process row: {row}, with error: {e}")

    with open(csv_output, "w", newline="") as outcf:
        writer = csv.writer(outcf)
        writer.writerow(["Filename", "Fragment [s]", "Mean Time [s]", "RTF"])
        for (filename, fragment) in sorted(agg.keys()): # Sort by filename and fragment length for consistent, deterministic output order
            runs = agg[(filename, fragment)]
            mean_time = sum(runs) / len(runs)
            rtf = mean_time / float(fragment)
            writer.writerow([filename, f"{fragment}", f"{mean_time:.6f}", f"{rtf:.6f}"])
            print(f"agg saved: {filename}, fragment: {fragment}s, mean: {mean_time:.6f}s, rtf: {rtf:.6f}")
    

# === Main Execution Block ===
if __name__ == "__main__":
    prefix = ""
    model = "scnet"
    db_path = None
    
    parser = argparse.ArgumentParser(description="Separation Speed Test")
    parser.add_argument('--prefix', type=str, default=prefix, help='Filename prefix for result files')
    parser.add_argument('--model', type=str, default=model, help='Model to test ("scnet" or "dttnet")')
    parser.add_argument('--db', type=str, default=db_path, help='Path to database directory')
    args = parser.parse_args()

    db_path = get_database(args.db)
    warmup_separation()
    csv_client, csv_model, agg_csv_client, agg_csv_model = get_result_files(args.prefix)
    results_client, results_model = test_separation_speed(args.model, args.prefix)
    results_to_csv(csv_client, results_client)
    results_to_csv(csv_model, results_model)
    aggregate_results_to_csv(csv_client, agg_csv_client)
    aggregate_results_to_csv(csv_model, agg_csv_model)