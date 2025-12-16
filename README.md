# music_source_separation_service
music_source_separation_service is a lightweight FastAPI service for performing source separation on uploaded audio files and returning the extracted stems: vocals, drums, bass, and other.

## features
* uploading audio files in different formats
* automatic audio validation and preprocessing
* SCNet and DTTNet inference workers for audio stem separation
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
4. configure the **scnet_worker**, **dttnet_worker**, and **main** ports in the following config files:
- app/workers/scnet/**scnet01_config.yaml**
```yaml
worker_id: scnet01
model_type: scnet
worker_address: scnet01:8101 # configure worker's port
main_address: main:8000 # configure main's port
```
- app/workers/dttnet/**dttnet01_config.yaml**
```yaml
worker_id: dttnet01
model_type: dttnet
worker_address: dttnet01:8201 # configure worker's port
main_address: main:8000 # configure main's port
```
5. run the **music source separation service**
```bash
python run_local.py
```
6. (optional) run each service individually
- run the **main** service
```bash
uvicorn app.main:app --port <MAIN_PORT_FROM_CONFIG>
```
- run the **scnet_worker** service (in different terminal)
```bash
uvicorn app.workers.scnet.scnet_worker:app --port <SCNET_WORKER_PORT_FROM_CONFIG>
```
- run the **dttnet_worker** service (in different terminal)
```bash
uvicorn app.workers.dttnet.dttnet_worker:app --port <DTTNET_WORKER_PORT_FROM_CONFIG>
```


## usage
open the interactive api docs:
```
https://127.0.0.1:<MAIN_PORT>/docs
```
use following endpoints:
* **/upload:** submit a model type (**scnet** or **dttnet**) and upload your audio file
* **/download:** provide a **file_id** or **download_url** to download separated stems


## output stems
service should generate:
* vocals.wav
* other.wav
* bass.wav
* drums.wav


## docker
1. make sure **docker** is installed
2. build the images
```bash
docker compose build
```
3. configure the **scnet_worker**, **dttnet_worker**, and **main** ports in the **docker-compose.yml**
```yml
ports:
    - "8000:8000" # Change left for different port on host machine for main
    - "8101:8101" # Change left for different port on host machine for scnet_worker
    - "8201:8201" # Change left for different port on host machine for dttnet_worker
```
4. start the environment
```bash
docker compose up
```
5. (optional dev-mode) run the environment mounting codebase enabling auto-reload on saving changes
* adjust ports in **docker-compose.dev.yml**
```yml
command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] # '--port' should match the one in docker-compose.yml
command: ["uvicorn", "workers.scnet.scnet_worker:app", "--host", "0.0.0.0", "--port", "8101", "--reload"]
command: ["uvicorn", "workers.dttnet.dttnet_worker:app", "--host", "0.0.0.0", "--port", "8201", "--reload"]  
```
* start the environment in **dev-mode**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```


## tests
1. activate the virtual environment (here: **.venv**)
```bash
source venv/bin/activate # linux/macOS
venv\Scripts\Activate.ps1 # windows
```
2. make sure all services are running (**locally** or via **docker**)
### functional tests
3. run functional tests using **pytest**
```bash
pytest -v  #'-s' for enabling debugging prints 
```

### performance tests
#### test source separation quality
3. download the **MUSDB18-HQ** _testing subset_ (uncompressed version), e.g. via: https://zenodo.org/records/3338373
4. set the path to the MUSDB18-HQ directory in **test_separation_quality.py**
```python
mus = musdb.DB(root="<YOUR/DIRECTORY>", is_wav=True, subsets="test")
```
5. run the **separation quality test**
```bash
python ./tests/performance/test_separation_quality.py --prefix <PREFIX> --model <MODEL>
```
##### arguments
- **--prefix**: filename prefix for the result files
- **--model**: model to test ('scnet' or 'dttnet')
6. (optional) specify default parameters in **test_separation_quality.py** to avoid passing arguments every time
```python
# === Main Execution Block ===
if __name__ == "__main__":
    prefix = ""
    model = "scnet"
```
#### test source separation speed
3. run the **separation speed test**
```bash
python ./tests/performance/test_separation_quality.py --prefix <PREFIX> --model <MODEL> --db <DB_DIRECTORY>
```
##### arguments
- **--prefix**: filename prefix for the result files
- **--model**: model to test ('scnet' or 'dttnet')
- **--db**: directory where the provided database will be downloaded or where your own existing one is stored (default: `./tests/`)
4. (optional) specify the default parameters in **test_separation_speed.py** to avoid passing arguments every time
```python
# === Main Execution Block ===
if __name__ == "__main__":
    prefix = ""
    model = "scnet"
    db_path = ""
```
