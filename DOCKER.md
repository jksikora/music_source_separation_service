# docker instructions
## run the service in production-mode
run the services with logs printed directly to the terminal
```bash
docker compose up
```

run the services without logs (in the background)
```bash
docker compose up -d
```

rebuild all images layer by layer and show which steps use cache / which execute during image building
```bash
docker compose build --progress=plain 
```

force a rebuild of every layer and run 
```bash
docker compose up --build
```


## run the service in dev-mode
enable dynamic service reload on code changes without rebuilding the images
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```


## run the containers separately
force a rebuild of every layer for a single service and run accordingly
```bash
docker compose up --build main
docker compose up --build scnet01
docker compose up --build dttnet01
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
docker compose logs -f main
docker compose logs -f scnet01
docker compose logs -f dttnet01
```

list compose services and images related to them
```bash
docker compose images
```



## close the service
stop the stack, remove containers and the default network:
```bash
docker compose down
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