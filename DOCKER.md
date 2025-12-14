# docker instructions
## run the service in the production-mode
build images
```bash
docker compose build
```
run the services
```bash
docker compose up
```
run the services without logs (in the background)
```bash
docker compose up -d
```
(after change in code) force a rebuild of changed layers and run 
```bash
docker compose up --build
```
destroy containers
```bash
docker compose down
```

## run the service in development-mode
enable dynamic service reload on code changes without rebuilding the images by mounting the codebase
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## run the containers separately
Force a rebuild of changed layers for a single service and start it
```bash
docker compose up --build --no-deps main
docker compose up --build --no-deps scnet01
docker compose up --build --no-deps dttnet01
```

## follow logs
show logs from all services defined in the compose file
```bash
docker compose logs -f
docker compose logs -f main # for a single main service
docker compose logs -f scnet01 # for a single scnet01 service
docker compose logs -f dttnet01 # for a single dttnet01 service
```

## verify services
list running containers and their status
```bash
docker compose ps
```
list compose services and images related to them
```bash
docker compose images
```

## build and run a single image manually
build and run the `main` service image:
```bash
docker build -f Dockerfile.main -t mss:main . # main service
docker run --rm -p 8000:8000 --name mss_main mss:main
```
build and run the `scnet01` service image:
```bash
docker build -f Dockerfile.scnet -t mss:scnet01 . # scnet service
docker run --rm -p 8101:8101 --name mss_scnet01 mss:scnet01
```
build and run the `dttnet01` service image:
```bash
docker build -f Dockerfile.dttnet -t mss:dttnet01 . # dttnet01 service
docker run --rm -p 8201:8201 --name mss_dttnet01 mss:dttnet01
```

**[warning] when running containers manually, inter-service communication via docker compose networking is not available**