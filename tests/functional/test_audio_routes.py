from app.services.storage import storage
from unittest.mock import patch
import pytest, io, torchaudio, asyncio, time

# --- Endpoint tests ---

# === Test upload_audio endpoint ===
@pytest.mark.parametrize("sample_audio_file", ["WAV", "FLAC"], indirect=True) # Runs test for both "WAV" and "FLAC" formats; indirect=True tells pytest to pass the param to the fixture not the test function
def test_upload_audio(client, sample_audio_file):
    """Testing upload_audio endpoint for successful stem extraction
    1. Check the response status code is 200
    2. Check the response JSON contains expected stems and fields"""

    audio_bytes, _, format = sample_audio_file
    files = {'file': (f'test.{format.lower()}', audio_bytes, f'audio/{format.lower()}')} #Preparing file for upload (name, content, mime type)
    upload_response = client.post("/upload_audio", files=files) #Making POST request to the upload_audio endpoint
    assert upload_response.status_code == 200, f"Unexpected status code: {upload_response.status_code}" #Check if response status code is 200
    data = upload_response.json() #Parsing JSON response (converts response body to Python dict)
    
    expected_stems = {"vocals", "drums", "bass", "other"}
    for stem in expected_stems:
        assert stem in data, f"Stem '{stem}' not found in response" #Check if each expected stem is in the result
        stem_info = data[stem]
        for field in ["file_id", "filename", "download_url"]:
            assert field in stem_info, f"Field '{field}' missing in stem '{stem}' info" #Check if each expected field is in the stem info


# === Test download_audio endpoint ===
@pytest.mark.parametrize("sample_audio_file", ["WAV", "FLAC"], indirect=True) # Runs test for both "WAV" and "FLAC" formats; indirect=True tells pytest to pass the param to the fixture not the test function
def test_download_audio(client, sample_audio_file):
    """Testing download_audio endpoint for successful stem extraction
    1. Check the response status code for upload and download is 200
    2. Check the response content type is audio/wav
    3. Check the response content is not empty
    4. Check the response content is a valid audio file that can be loaded with torchaudio
    5. Check the downloaded audio's sample rate matches the original"""

    audio_bytes, sample_rate, format = sample_audio_file
    files = {'file': (f'test.{format.lower()}', audio_bytes, f'audio/{format.lower()}')} #Preparing file for upload (name, content, mime type)
    upload_response = client.post("/upload_audio", files=files)
    assert upload_response.status_code == 200, f"Unexpected status code: {upload_response.status_code}" #Check if upload response status code is 200
    data = upload_response.json()

    for stem_name, stem_info in data.items(): # Check if all stems can be downloaded and are valid audio files
        file_id = stem_info["file_id"]
        download_response = client.get(f"/download_audio/{file_id}") # Making GET request to the download_audio endpoint with the stem's file_id
        assert download_response.status_code == 200, f"Unexpected status code for {stem_name}: {download_response.status_code}" # Check if download response status code is 200
        assert download_response.headers["content-type"].lower() == "audio/wav", f"Unexpected content type for {stem_name}: {download_response.headers['content-type']}" # Check if content type is audio/wav (case insensitive)
        assert int(download_response.headers["content-length"]) > 0, f"Downloaded file for {stem_name} is empty" # Check if the response content is not empty 

        buffer = io.BytesIO(download_response.content)
        downloaded_waveform, downloaded_sample_rate = torchaudio.load(buffer) # Load audio from the response content and verify if it's valid with torchaudio
        assert len(downloaded_waveform) > 0 and downloaded_waveform is not None, f"Downloaded audio for {stem_name} is empty or invalid" # Check if the waveform is not empty and is valid
        assert downloaded_sample_rate == sample_rate and downloaded_sample_rate is not None, f"Sample rate mismatch for {stem_name}: expected {sample_rate}, got {downloaded_sample_rate}" # Check if the downloaded audio's sample rate matches the original


# === Test upload_audio endpoint with invalid file ===
def test_upload_audio_invalid_file(client):
    """Testing upload_audio endpoint with invalid file
     1. Check the response status code is 400
     2. Check the response JSON contains the expected error message"""
    
    files = {'file': ('test.txt', b'Invalid file.', 'text/plain')} #Preparing a non-audio file for upload
    response = client.post("/upload_audio", files=files) #Making POST request with an invalid file
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert data["detail"] == "Invalid audio file", f"Unexpected error message: {data['detail']}" #Check that the error message is as expected


# === Test upload_audio endpoint with concurrent uploads ===
@pytest.mark.asyncio
async def test_upload_audio_concurrent_uploads(async_client, sample_audio_file):
    """Testing concurrent uploads to check if the system handles multiple requests properly
    1. Send two uploads at the same time
    2. Check if one upload succeeds and the other fails with expected error message"""
    audio_bytes, _, _ = sample_audio_file
    files = {'file': ('test.wav', audio_bytes, 'audio/wav')}
    task1 = async_client.post("/upload_audio", files=files)
    task2 = async_client.post("/upload_audio", files=files)
    response1, response2 = await asyncio.gather(task1, task2) # Send two upload requests concurrently

    statuses = {response1.status_code, response2.status_code}
    assert statuses == {200, 503}, f"Unexpected statuses codes: expected one 200 and one 503, got {statuses}"

    success_response = response1 if response1.status_code == 200 else response2 # The successful one should have all expected stems
    success_data = success_response.json()
    expected_stems = {"vocals", "drums", "bass", "other"}
    for stem in expected_stems:
        assert stem in success_data, f"Stem '{stem}' not found in successful response"

    fail_response = response1 if response1.status_code == 503 else response2 # The failed one should have no_available_workers error
    fail_data = fail_response.json()
    assert fail_data['detail'] == "No available SCNet workers", f"Unexpected error message: {fail_data['detail']}"


# === Test download_audio endpoint with invalid file_id ===
def test_download_audio_invalid_id(client):
    """Testing download_audio endpoint with invalid file_id
     1. Check the response status code is 404
     2. Check the response JSON contains the expected error message"""
    
    invalid_file_id = "non_existent_file_id"
    response = client.get(f"/download_audio/{invalid_file_id}") #Making GET request with an invalid file_id
    assert response.status_code == 404, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert data["detail"] == "File not found", f"Unexpected error message: {data['detail']}" #Check that the error message is as expected

# --- Service tests ---

# === Test storage cleanup function ===
@pytest.mark.asyncio
async def test_storage_cleanup(sample_audio_file_in_storage):
    """Testing storage cleanup function to ensure expired files are deleted
    1. Check that a saved file exists
    2. Check that after expiration time the file is deleted by simulating expiration and triggering cleanup"""

    file_id = sample_audio_file_in_storage
    assert await storage.exists(file_id), "File is expected to exist in storage before expiration"

    original_time = time.time()
    expired_time = original_time + 10 * 60 + 1

    with patch('time.time', return_value=expired_time): # Mock time.time to simulate expiration (10 minutes + 1 second later)
        await storage._cleanup() # Cleanup must be triggered manually here because it is called during save operation

    assert not await storage.exists(file_id), "File is expected to be deleted from storage after expiration"