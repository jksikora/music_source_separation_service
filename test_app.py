from fastapi.testclient import TestClient
from fastapi import UploadFile, HTTPException
import pytest
import torchaudio
import io
import soundfile as sf
import numpy as np
from app import (
    app, 
    audiofile_verification, 
    music_source_separation,
    convert_to_audio_buffer,
    buffer_generator, 
    storage
)

# --- Fixtures ---

# For each test create a different TestClient to interact with the FastAPI app endpoints in tests.
@pytest.fixture
def client():
    return TestClient(app)

# Clear the in-memory storage before and after each test to ensure test isolation
@pytest.fixture(autouse=True) #Automatically use this fixture for all tests
def clear_storage():
    storage.clear()
    yield
    storage.clear()

# For each test create a different sample audio file that can be used to simulate file uploads
@pytest.fixture
def sample_audio_file():
    fs=44100
    waveform = np.random.randn(fs).astype(np.float32) * 0.1 #1 seconds of random audio noise (soundfile and torchaudio expect float32 format)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, fs, format='WAV') #waveform instead of waveform.T.numpy(), because it's already a numpy array
    buffer.seek(0)
    
    return buffer.read(), fs

# --- Endpoint tests ---

# Testing upload_audio endpoint
# Goals:
# 1. Check the response status code is 200
# 2. Check the response JSON contains expected stems and fields
def test_upload_audio_returns_stems(client, sample_audio_file):
    audio_bytes, _ = sample_audio_file
    files = {'file': ('test.wav', audio_bytes, 'audio/wav')} #Preparing file for upload (name, content, mime type)
    response = client.post("/upload_audio/", files=files) #Making POST request to the upload_audio endpoint
    
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}" #Check if response status code is 200
    data = response.json() #Parsing JSON response (converts response body to Python dict)
    result = data["result"] #Extracting the 'result' field from the response JSON
    
    expected_stems = {"vocals", "drums", "bass", "other"}
    for stem in expected_stems:
        assert stem in result, f"Stem '{stem}' not found in response" #Check if each expected stem is in the result
        stem_info = result[stem]
        for field in ["file_id", "filename", "download_url"]:
            assert field in stem_info, f"Field '{field}' missing in stem '{stem}' info" #Check if each expected field is in the stem info

# Testing download_audio endpoint
# Goals:
# 1. Check the response status code for upload and download is 200
# 2. Check the response content type is audio/wav
# 3. Check the response content is not empty
# 4. Check the response content is a valid audio file that can be loaded with torchaudio
# 5. Check the waveform content matches the original uploaded audio
# 6. Check the downloaded audio's sample rate matches the original
def test_download_audio_returns_valid_file(client, sample_audio_file):
    audio_bytes, fs = sample_audio_file
    files = {'file': ('test.wav', audio_bytes, 'audio/wav')}
    upload_response = client.post("/upload_audio/", files=files)

    assert upload_response.status_code == 200, f"Unexpected status code: {upload_response.status_code}" #Check if upload response status code is 200
    data = upload_response.json()
    result = data["result"]

    first_stem_info = next(iter(result.values())) #Get info of the first stem
    file_id = first_stem_info["file_id"]

    download_response = client.get(f"/download_audio/{file_id}") #Making GET request to the download_audio endpoint with the stem's file_id
    assert download_response.status_code == 200, f"Unexpected status code: {download_response.status_code}" #Check if download response status code is 200
    assert download_response.headers["content-type"].lower() == "audio/wav", f"Unexpected content type: {download_response.headers['content-type']}" #Check if content type is audio/wav (case insensitive)
    assert int(download_response.headers["content-length"]) > 0, "Downloaded file is empty" #Check that the response content is not empty 

    buffer = io.BytesIO(download_response.content)
    waveform, sample_rate = torchaudio.load(buffer) #Load audio from the response content and verify it's valid with torchaudio
    original_waveform, _ = torchaudio.load(io.BytesIO(audio_bytes))
    assert np.array_equal(waveform.numpy(), original_waveform.numpy()), "Waveform content mismatch"  #Check that the waveform matches the original waveform
    assert sample_rate == fs, f"Sample rate mismatch: expected {fs}, got {sample_rate}" #Check that the downloaded audio's sample rate matches the original

# Testing upload_audio endpoint with invalid file
# Goals:
# 1. Check the response status code is 400
# 2. Check the response JSON contains the expected error message
def test_upload_audio_invalid_file(client):
    files = {'file': ('test.txt', b'Invalid file.', 'text/plain')} #Preparing a non-audio file for upload
    response = client.post("/upload_audio/", files=files) #Making POST request with an invalid file
    
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert data["detail"] == "Invalid audio file", "Unexpected error message" #Check that the error message is as expected

# Testing download_audio endpoint with invalid file_id
# Goals:
# 1. Check the response status code is 404
# 2. Check the response JSON contains the expected error message
def test_download_audio_invalid_id(client):
    invalid_file_id = "non_existent_file_id"
    response = client.get(f"/download_audio/{invalid_file_id}") #Making GET request with an invalid file_id
    
    assert response.status_code == 404, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert data["detail"] == "File not found", "Unexpected error message" #Check that the error message is as expected

# --- Unit tests ---

# Testing audiofile_verification function
# Goals:
# 1. Check that a valid audio file is processed correctly
# 2. Check that the function returns dictionary with expected keys and types
# 3. Check filename and sample rate match the input file
# 4. Check that the waveform has data
# 5. Check that an invalid audio file raises the appropriate exception
# 6. Check that the exception has the correct status code and message
# 7. Check the response JSON contains the expected error message
@pytest.mark.asyncio
async def test_audiofile_verification_valid_file(sample_audio_file):
    audio_bytes, fs = sample_audio_file
    upload_file = UploadFile(filename="test.wav", file=io.BytesIO(audio_bytes))
    
    result = await audiofile_verification(upload_file)

    assert isinstance(result, dict), "Result is not a dictionary" #Check that result is a dictionary
    for key in ["file", "waveform", "sample_rate"]:
        assert key in result, f"Key '{key}' missing in result" #Check that all expected keys are present
    
    #assert isinstance(result["waveform"], torch.Tensor)
    assert result["file"].filename == "test.wav", "Filename mismatch" #Check that the filename matches
    assert result["sample_rate"] == fs, "Sample rate mismatch" #Check that the sample rate matches
    assert result["waveform"].shape[1] > 0, "Waveform has no data"  #Check that the waveform has data

@pytest.mark.asyncio
async def test_audiofile_verification_invalid_file():
    upload_file = UploadFile(filename="test.txt", file=io.BytesIO(b'Invalid file.'))
    
    with pytest.raises(HTTPException) as exc_info: #Expecting an HTTPException to be raised for invalid file
        await audiofile_verification(upload_file)
    
    assert exc_info.value.status_code == 400, "Unexpected status code" #Check that the status code is 400
    assert exc_info.value.detail == "Invalid audio file", "Unexpected error message" #Check that the error message is as expected

# Testing music_source_separation function
# Goals:
# 1. Check that the function returns a dictionary with expected stems
# 2. Check that each stem contains expected keys and types
# 3. Check that the stems are stored in the in-memory storage with correct data
# 4. Check that the download URL is correct
# 5. Check that the stored waveform matches the stem audio
def test_music_source_separation(sample_audio_file):
    audio_bytes, _ = sample_audio_file
    upload_file = UploadFile(filename="test.wav", file=io.BytesIO(audio_bytes))
    
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)
    audiofile_dict = {
        "file": upload_file,
        "waveform": waveform,
        "sample_rate": sample_rate
    }

    result = music_source_separation(audiofile = audiofile_dict)

    assert isinstance(result, dict), "Result is not a dictionary" #Check that result is a dictionary
    expected_stems = {"vocals", "drums", "bass", "other"}
    assert set(result.keys()) == expected_stems, "The result does not contain the expected stems" #Check that there is no missing or extra stems

    expected_keys = {"file_id", "filename", "download_url"}
    for stem_name, stem_info in result.items(): #Tulple unpacking to get stem name and its info
        assert set(stem_info.keys()) == expected_keys, f"Stem '{stem_name}' does not contain the expected keys" #Check that each stem info has all expected keys
        
        file_id = stem_info["file_id"] #Get the file_id of the stem
        assert file_id in storage, f"Stem '{stem_name}' with file_id '{file_id}' not found in storage" #Check that the stem is stored in memory

        stored_filename, stored_waveform, stored_fs = storage[file_id] #Retrieve stored data from in-memory storage
        assert stored_filename == stem_info["filename"], f"Filename mismatch for stem '{stem_name}'" #Check that the stored filename matches the stem filename
        assert stored_fs == sample_rate, f"Sample rate mismatch for stem '{stem_name}'" #Check that the stored sample rate matches the stem sample rate
        assert stem_info["download_url"] == f"/download_audio/{file_id}", f"Download URL mismatch for stem '{stem_name}'"  #Check that the download URL is correct
        #assert torchaudio.equal(stored_waveform, stem_info["audio"])  #Check that stored waveform matches the stem audio
        assert np.array_equal(stored_waveform.numpy(), waveform.numpy()), f"Waveform content mismatch for stem '{stem_name}'"  #Check that stored waveform matches the stem audio

# Testing convert_to_audio_buffer function
# Goals:
# 1. Check that the function returns an io.BytesIO object
# 2. Check that the size is greater than 0
# 3. Check that the filename ends with .wav
# 4. Check that the size matches the actual buffer length
# 5. Check that the saved buffer contains valid audio data that can be loaded with torchaudio
# 6. Check that the loaded sample rate matches the original sample rate
# 7. Check that the loaded waveform is not empty
# 8. Check that the loaded waveform matches the original waveform
def test_convert_to_audio_buffer(sample_audio_file):
    audio_bytes, _ = sample_audio_file
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)

    filename = "test.wav"
    save_buffer, size, filename = convert_to_audio_buffer(waveform, sample_rate, filename)

    assert isinstance(save_buffer, io.BytesIO), "Result is not an io.BytesIO object" #Check that the result is an io.BytesIO object
    assert size > 0, "Size is not greater than 0"  #Check that the size is greater than 0 
    assert filename.lower().endswith(".wav"), "Filename does not end with .wav"  #Check that filename ends with .wav
    assert len(save_buffer.getvalue()) == size, "Size does not match buffer length"  #Check that size matches the actual buffer length

    save_buffer.seek(0) #Reset buffer pointer to the beginning just in case
    loaded_waveform, loaded_sample_rate = torchaudio.load(save_buffer) #Check if the saved buffer contains valid audio data
    
    assert loaded_sample_rate == sample_rate, "Sample rate mismatch after loading from buffer" #Check that the loaded sample rate matches the original sample rate
    assert loaded_waveform.shape[1] > 0, "Loaded waveform has no data"  #Check that the loaded waveform has data
    #assert torchaudio.equal(loaded_waveform, waveform)  #Check that loaded waveform matches the original waveform
    assert np.array_equal(loaded_waveform.numpy(), waveform.numpy()), "Waveform content mismatch"  #Check that loaded waveform matches the original waveform

# Testing buffer_generator function
# Goals:
# 1. Check that the generator yields chunks of the specified size
# 2. Check that the reassembled bytes match the original bytes
# 3. Check that all chunks except possibly the last are of the specified size
def test_buffer_generator(sample_audio_file):
    audio_bytes, _ = sample_audio_file
    buffer = io.BytesIO(audio_bytes)
    chunk_size = 1024

    chunks = list(buffer_generator(buffer, chunk_size=chunk_size)) #buffer_generator is a generator, so we convert it to a list to read all chunks at once
    reassembled = b"".join(chunks) #Reassemble the chunks back into a single bytes object

    buffer.seek(0)
    original = buffer.read() #Read the original bytes from the buffer

    assert reassembled == original, "Reassembled bytes do not match original bytes"  #Check that reassembled bytes match the original bytes
    assert all(len(chunk) <= chunk_size for chunk in chunks[:-1]), "One or more chunks exceed the specified chunk size"  #Check that all chunks except possibly the last are of the specified size

# --- Integration tests ---

#Integration of both convert_to_audio_buffer and buffer_generator functions - testing end-to-end conversion and streaming
# Goals:
# 1. Check that convert_to_audio_buffer returns valid buffer, size, and filename
# 2. Check that buffer_generator correctly streams the buffer in chunks with specified size
# 3. Check that the reassembled streamed data matches the original buffer data
# 4. Check that the streamed data can be loaded back into a valid audio waveform
# 5. Check that the loaded sample rate matches the original sample rate 
# 6. Check that the loaded waveform matches the original waveform
def test_integration_convert_to_audio_buffer_and_buffer_generator(sample_audio_file):
    #Testing convert_to_audio_buffer part
    audio_bytes, _ = sample_audio_file
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)

    filename = "test.wav"
    save_buffer, size, filename = convert_to_audio_buffer(waveform, sample_rate, filename)

    assert isinstance(save_buffer, io.BytesIO), "Result is not an io.BytesIO object" #Check that the result is an io.BytesIO object
    assert size > 0, "Size is not greater than 0"  #Check that the size is greater than 0 
    assert filename.lower().endswith(".wav"), "Filename does not end with .wav"  #Check that filename ends with .wav
    assert len(save_buffer.getvalue()) == size, "Size does not match buffer length"  #Check that size matches the actual buffer length

    #Testing buffer_generator part
    chunk_size = 1024
    chunks = list(buffer_generator(save_buffer, chunk_size=chunk_size)) #buffer_generator is a generator, so we convert it to a list to read all chunks at once
    reassembled = b"".join(chunks) #Reassemble the chunks back into a single bytes object

    assert reassembled == save_buffer.getvalue(), "Reassembled bytes do not match original bytes"  #Check that reassembled bytes match the original bytes
    assert all(len(chunk) <= chunk_size for chunk in chunks[:-1]), "One or more chunks exceed the specified chunk size"  #Check that all chunks except possibly the last are of the specified size

    reassembled_buffer = io.BytesIO(reassembled)
    loaded_waveform, loaded_sample_rate = torchaudio.load(reassembled_buffer)

    assert loaded_sample_rate == sample_rate, "Sample rate mismatch after streaming" #Check that the loaded sample rate matches the original sample rate
    #assert torchaudio.equal(loaded_waveform, waveform), "Waveform content mismatch"
    assert np.array_equal(loaded_waveform.numpy(), waveform.numpy()), "Waveform content mismatch" #Check that loaded waveform matches the original waveform