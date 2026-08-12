# MLOps Assignment – CIFAR‑10 Classifier

## Overview
This project implements a complete MLOps pipeline for training and serving a CIFAR‑10 image classifier. It follows the assignment rubric and runs entirely inside Docker using a **single Docker image** for both training and serving. Docker‑Compose orchestrates the workflow locally, and Kubernetes manifests are provided for deployment to a cluster.

## Architecture
```mermaid
flowchart TD
    subgraph DockerImage[Docker Image: mlops-assgn3:latest]
        A[src/]
        B[configs/]
        C[requirements.txt]
    end
    DockerCompose -->|training| DockerImage
    DockerCompose -->|serving| DockerImage
    DockerImage -->|mount| data_volume[(/app/data)]
    DockerImage -->|mount| ckpt_volume[(/app/checkpoints)]
    classDef orange fill:#ffebcc,stroke:#ffa500,stroke-width:2px;
    class DockerCompose orange;
```

## Quick Start (Docker Compose)
```bash
# Build the image and start training (runs once and exits)
docker compose up --build training

# After training finishes, start the serving API
docker compose up -d serving

# Verify the service
curl http://localhost:8080/health   # should return {"status":"healthy"}
```

## Repository Structure
```
mlops-assgn3/
├─ .gitignore
├─ Dockerfile               # single image
├─ docker-compose.yml
├─ requirements.txt
├─ setup.sh
├─ README.md                # you are reading it :) 
├─ src/
│  ├─ dataset.py           # data loader utilities
│  ├─ model.py             # model definition
│  ├─ train.py             # training script
│  └─ serve.py             # FastAPI serving app
├─ configs/
│  ├─ training_config.yaml
│  └─ serving_config.yaml
└─ k8s/                     # optional Kubernetes manifests
```

## Linting & CI
The CI pipeline runs `flake8` and `black` on every push/PR. See `.github/workflows/ci.yml`.
