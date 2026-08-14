# MLOps Assignment – CIFAR‑10 Classifier

## Overview
This project implements a complete MLOps pipeline for training and serving a CIFAR‑10 image classifier using PyTorch, Docker, and Kubernetes. It supports multi-stage container builds, reproducible training with checkpoint persistence, and scalable inference with health probes and HPA autoscaling.

```mermaid
flowchart TD
    subgraph Cluster[Kubernetes Cluster: ml-training]
        subgraph Training[Training Phase]
            T[Training Job] -->|writes checkpoint| PVC[PVC: /app/checkpoints]
            T -->|reads/caches data| DataVol[Data PVC: /app/data]
            CM1[ConfigMap: training-config] -->|mounted config| T
        end
        subgraph Serving[Serving Phase]
            S[Model Serving Pods (2 replicas)] -->|reads checkpoint| PVC
            CM2[ConfigMap: training-config] -->|mounted config| S
            S -->|exposes port 8080| Service["Service (ClusterIP: 80)"]
            HPA[HorizontalPodAutoscaler] -->|scales| S
        end
    end
    DockerImageTrain[Docker Image: mlops-train:v1] -->|used by| T
    DockerImageServe[Docker Image: mlops-serve:v1] -->|used by| S
    classDef orange fill:#ffebcc,stroke:#ffa500,stroke-width:2px;
    class DockerImageTrain,DockerImageServe orange;
```

## Quick Start (Docker)

```bash
# 1. Build training and serving images
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# 2. Run containerized training with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# 3. Run model serving
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# 4. Test prediction endpoint
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## Quick Start (Kubernetes)

```bash
# 1. Create namespace and configuration
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc-data.yaml
kubectl apply -f k8s/pvc-checkpoints.yaml

# 2. Run training Job
kubectl apply -f k8s/training-job.yaml

# 3. Deploy model serving and autoscaler
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Port-forward for testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training

# 5. Test health and predictions
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

> **Note**: You can also deploy all manifests using Kustomize: `kubectl apply -k k8s/`

## Repository Structure
```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker-build.yml
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
├── configs/
│   ├── training_config.yaml
│   └── serving_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml
│   ├── pvc-data.yaml
│   └── pvc-checkpoints.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Linting & CI
The CI pipeline runs `black`, `flake8`, and `pytest` on every push/PR. See [.github/workflows/ci.yml](file:///.github/workflows/ci.yml).
