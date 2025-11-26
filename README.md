# music_source_separation_service
music_source_separation_service is a lightweight FastAPI service for performing source separation on uploaded audio files and returning the extracted stems: vocals, drums, bass, and other.
## features
* uploading audio files in different formats
* automatic audio validation and preprocessing
* (for now only) SCNet inference worker for stem separation
* downloading separated stems

## installation
1. clone the repository
```bash
git clone https://github.com/jksikora/music_source_separation_service.git
cd music_source_separation_service
```
2. create **.venv** with python3.10
```bash
python3.10 -m venv venv
source venv/bin/activate # linux/macOS
venv\Scripts\activate # windows
```
3. install **requirements.txt**
```bash
pip install -r requirements.txt
```
4. configurate **scnet_worker** and **main** ports in the file: app/workers/scnet/**scnet01_config.yaml**
```yaml
worker_id: scnet01
model_type: scnet
worker_address: 127.0.0.1:8100 # configurate worker_address
main_address: 127.0.0.1:8000 # configurate main_address
```
5. run the **main** service
```bash
uvicorn app.main:app --port <MAIN_PORT_FROM_CONFIG>
```
6. run the **scnet_worker** service (in another terminal)
```bash
uvicorn app.workers.scnet_worker:app --port <WORKER_PORT_FROM_CONFIG>
```

## usage
open the interactive api docs:
```
https://127.0.0.1:<MAIN_PORT>/docs
```
use following endpoints:

* **/upload:** upload your audio file
* **/download:** download separated stems

## output stems
service should generate:
* vocals.wav
* other.wav
* bass.wav
* drums.wav
