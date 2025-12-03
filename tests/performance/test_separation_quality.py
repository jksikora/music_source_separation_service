import soundfile as sf
import numpy as np
import musdb, museval, io, csv, os, yaml, httpx

# --- Separation Quality Benchmark Test ---

# === Load MUSDB18-HQ Dataset ===
mus = musdb.DB(root="/mnt/d/studia/praca_inzynierska/musdb18hq", is_wav=True, subsets="test")


# === Main Address Loaded from YAML File ===
with open("workers/scnet/scnet1_config.yaml", "r") as f:
    config = yaml.safe_load(f)
main_address = config["main_address"]  # e.g., "127.0.0.1:8000"; If tests are run on the host (not inside Docker) the compose service name `main:8000` is not resolvable from the host

if isinstance(main_address, str) and ":" in main_address: # When the config contains the compose service name, 
    _, port = main_address.split(":", 1) # translate it to the host address so tests can connect to the published port (localhost)
    main_address = f"127.0.0.1:{port}"


# === Create HTTP Client ===
client = httpx.Client(base_url=f"http://{main_address}", timeout=600.0) # Increased timeout to 600 seconds for long-running model inference


# === Prepare Results Directory and CSV Files for Separation Quality Test Results ===
results_dir = "tests/performance/scnet_results/separation_quality"
os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist; exist_ok=True avoids error if it already exists
csv_frames = os.path.join(results_dir, "docker_agg_frames_separation_quality_results.csv") # Complete path to aggregated frame-level separation quality results CSV file
csv_tracks = os.path.join(results_dir, "docker_agg_tracks_separation_quality_results.csv") # Complete path to aggregated track-level separation quality results CSV file

if not os.path.exists(csv_frames):
    with open(csv_frames, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Filename", "Stem", "SDR", "SIR", "SAR", "ISR"])

if not os.path.exists(csv_tracks):
    with open(csv_tracks, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Stem", "SDR", "SIR", "SAR", "ISR"])


# === Convert Audio to Bytes Helper Function ===
def _audio_to_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


# === Test Separation Quality Function ===
def test_separation_quality() -> None:
    results = museval.EvalStore() # Initialize EvalStore to hold separation quality results
    for track in mus.tracks:  # Limit to first n songs by adding [:n]
        try: # Added try-except to allow saving partial results
            print(f"\nstart processing file: {track.name}")
            
            input_audio_bytes = _audio_to_bytes(track.audio, track.rate) # Convert musdb mixture from numpy.ndarray to bytes
            files = {'file': (f'{track.name}.wav', input_audio_bytes, 'audio/wav')}
            upload_response = client.post("/upload_audio", files=files)

            if upload_response.status_code != 200:
                raise RuntimeError(f"unexpected status code from upload_audio endpoint response: {upload_response.status_code} for file: {track.name}")
            
            data = upload_response.json() # Get separation result metadata
            estimates = {}
            for stem_name, stem_data in data.items():
                file_id = stem_data['file_id']
                download_resp = client.get(f"/download_audio/{file_id}")
                if download_resp.status_code == 200:
                    buffer = io.BytesIO(download_resp.content)
                    output_waveform, _ = sf.read(buffer)
                    estimates[stem_name] = output_waveform # Store estimated stem waveform
                    print(f"estimated stem {stem_name} of file: {track.name} shape: {estimates[stem_name].shape}")
                    print(f"reference stem {stem_name} of file: {track.name} shape: {track.targets[stem_name].audio.shape}")
                
                else:
                    raise RuntimeError(f"unexpected status code from download_audio endpoint response: {download_resp.status_code} for {stem_name} of file: {track.name}")
            
            scores = museval.eval_mus_track(track, estimates, output_dir=results_dir) # Evaluate separation quality track by track and save results to json files
            results.add_track(scores) # Add scores to EvalStore to have overall results in pandas dataframe and to aggregate metrics (median, mean / by frame, stem, tracks)

        except Exception as e: # If the server stops or any error occurs during measurement, save collected results
            print(f"evaluation failed for: {track.name} with error: {e}")
            
            if len(results.df) > 0:
                print("saving results after interruption")
                try:
                    results.save(os.path.join(results_dir, "docker_separation_quality_results.pandas")) # Save EvalStore results for later analysis
                    agg_frames_to_csv(csv_frames)
                    agg_tracks_to_csv(csv_tracks)
                except Exception as e:
                    print(f"failed to save partial results, error: {e}")
            else:
                print("no results to save after interruption")
            break

    results.save(os.path.join(results_dir, "docker_separation_quality_results.pandas")) # Save EvalStore results for later analysis


# === Save Overall Results Aggregated by Frames to CSV Function ===
def agg_frames_to_csv(csv_frames: str) -> None:
    results = museval.EvalStore() # Initialize EvalStore to hold separation quality results
    results_path = os.path.join(results_dir, "docker_separation_quality_results.pandas") # Path to saved EvalStore results
    results.load(results_path) # Load previously saved EvalStore results
    try:
        dataframe = results.agg_frames_scores() # Get aggregated frame-level scores (one metric per stem per track)
        with open(csv_frames, 'a', newline="") as cf:
            writer = csv.writer(cf)
            for track in dataframe.index.get_level_values("track").unique(): # Iterate over unique tracks
                track_df = dataframe.loc[track] # DataFrame for the specific track
                for stem in track_df.index.get_level_values("target").unique(): # Iterate over unique stems for the track
                    stem_df = track_df.loc[stem] # DataFrame for the specific stem
                    sdr = stem_df.loc["SDR"] # Median SDR for the stem
                    sir = stem_df.loc["SIR"] # Median SIR for the stem
                    sar = stem_df.loc["SAR"] # Median SAR for the stem
                    isr = stem_df.loc["ISR"] # Median ISR for the stem
                    writer.writerow([track, stem, sdr, sir, sar, isr]) # Write results to CSV

                    print(f"{track} {stem}: "f"SDR={sdr:.3f}, SIR={sir:.3f}, SAR={sar:.3f}, ISR={isr:.3f}")
        
    except Exception as e:
        print(f"saving overall results aggregated by frames to csv failed with error: {e}")


# === Save Overall Results Aggregated by Tracks to CSV Function ===
def agg_tracks_to_csv(csv_tracks: str) -> None:
    results = museval.EvalStore() # Initialize EvalStore to hold separation quality results
    results_path = os.path.join(results_dir, "docker_separation_quality_results.pandas") # Path to saved EvalStore results
    results.load(results_path) # Load previously saved EvalStore results
    try:
        dataframe = results.agg_frames_tracks_scores() # Get aggregated track-level scores (one metric per stem)
        with open(csv_tracks, 'a', newline="") as cf:
            writer = csv.writer(cf)
            for stem in dataframe.index.get_level_values("target").unique(): # Iterate over unique stems
                stem_df = dataframe.loc[stem] # DataFrame for the specific stem
                sdr = stem_df.loc["SDR"] # Median SDR for the stem
                sir = stem_df.loc["SIR"] # Median SIR for the stem
                sar = stem_df.loc["SAR"] # Median SAR for the stem
                isr = stem_df.loc["ISR"] # Median ISR for the stem
                writer.writerow([stem, sdr, sir, sar, isr]) # Write results to CSV

                print(f"{stem}: "f"SDR={sdr:.3f}, SIR={sir:.3f}, SAR={sar:.3f}, ISR={isr:.3f}")
        
    except Exception as e:
        print(f"saving overall results aggregated by tracks to csv failed with error: {e}")


# === Main Execution Block ===
if __name__ == "__main__":
    test_separation_quality()
    agg_frames_to_csv(csv_frames)
    agg_tracks_to_csv(csv_tracks)