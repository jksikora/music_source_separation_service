import soundfile as sf
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


# === Prepare Results Directory and CSV Files for Separation Quality Test Results ===
results_dir = "tests/performance/results"
os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist; exist_ok=True avoids error if it already exists
csv_frames = os.path.join(results_dir, "agg_frames_separation_quality_results.csv") # Complete path to aggregated by frames CSV file
csv_tracks = os.path.join(results_dir, "agg_tracks_separation_quality_results.csv") # Complete path to aggregated by tracks CSV file

if not os.path.exists(csv_frames):
    with open(csv_frames, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Filename", "Stem", "SDR", "SIR", "SAR", "ISR"])

if not os.path.exists(csv_tracks):
    with open(csv_tracks, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Stem", "SDR", "SIR", "SAR", "ISR"])


# === Convert Audio to Bytes Helper Function ===
def audio_to_bytes(audio, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


# === Test Separation Quality Function ===
def test_separation_quality():
    results = museval.EvalStore() # Initialize EvalStore to hold separation quality results
    for track in mus.tracks:  # Limit to first 1 for testing; remove [:1] for full test
        print(f"\nstart processing file: {track.name}")
        
        input_audio_bytes = audio_to_bytes(track.audio, track.rate) # Convert musdb mixture from numpy.ndarray to bytes
        files = {'file': (f'{track.name}.wav', input_audio_bytes, 'audio/wav')}
        upload_response = client.post("/upload_audio", files=files)

        if upload_response.status_code != 200:
            print(f"unexpected status code from upload_audio endpoint response: {upload_response.status_code} for file: {track.name}")
            continue # Skip to next track on failure
        
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
                print(f"unexpected status code from download_audio endpoint response: {download_resp.status_code} for {stem_name} of file: {track.name}")
            
        try:
            scores = museval.eval_mus_track(track, estimates, output_dir=results_dir) # Evaluate separation quality track by track and save results to json files
            results.add_track(scores) # Add scores to EvalStore to have overall results in pandas dataframe and to aggregate metrics (median, mean / by frame, stem, tracks)

        except Exception as e:
            print(f"evaluation failed for: {track.name} with error: {e}")
    
    results.save(os.path.join(results_dir, "separation_quality_results.pandas")) # Save EvalStore results for later analysis


# === Save Overall Results Aggregated by Frames to CSV Function ===
def agg_frames_to_csv(csv_frames: str):
    results = museval.EvalStore() # Initialize EvalStore to hold separation quality results
    results_path = os.path.join(results_dir, "separation_quality_results.pandas") # Path to saved EvalStore results
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
def agg_tracks_to_csv(csv_tracks: str):
    results = museval.EvalStore() # Initialize EvalStore to hold separation quality results
    results_path = os.path.join(results_dir, "separation_quality_results.pandas") # Path to saved EvalStore results
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

if __name__ == "__main__":
    #test_separation_quality()
    agg_frames_to_csv(csv_frames)
    agg_tracks_to_csv(csv_tracks)