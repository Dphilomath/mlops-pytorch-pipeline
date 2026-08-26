# Technical Challenges & Solutions Log

This document records the major engineering and operational challenges encountered during the setup, optimization, containerization, and Kubernetes deployment of the MLOps PyTorch pipeline, along with their root causes and resolutions.

---

## 1. PyTorch Docker Image Bloat (8.5 GB vs. 1.4 GB)

* **Symptom / Problem:**  
  Building the Docker images locally on macOS resulted in images of ~1.0 GB. However, building the same Dockerfiles on the remote Ubuntu server resulted in images exceeding **8.5 GB** per container (`mlops-pytorch-pipeline-training` and `mlops-pytorch-pipeline-serving`).

* **Root Cause Analysis:**  
  Standard `pip install torch` on Linux x86_64 fetches the default PyPI wheel, which bundles full NVIDIA CUDA 12, cuDNN, cuBLAS, and NCCL shared libraries (`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc.), adding ~7.5 GB of unused GPU dependencies into `site-packages`. macOS builds do not download CUDA wheels, which caused the local vs. remote disparity.

* **Resolution:**  
  Updated `Dockerfile.train`, `Dockerfile.serve`, and `Dockerfile` to pull CPU-only wheels directly from the PyTorch wheel repository:
  ```dockerfile
  RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
  ```
  Additionally, updated `.dockerignore` to exclude datasets (`data/`), model checkpoints (`checkpoints/`, `*.pt`), logs, and python build artifacts.  
  **Result:** Remote Docker image size dropped from **8.52 GB to 1.44 GB** (~83% size reduction).

---

## 2. Training Loop Deadlock / Hung at "Starting Epoch 1/10"

* **Symptom / Problem:**  
  When executing `docker compose up training`, the container logged `Starting epoch 1/10` and froze for over an hour without processing any data batches or showing progress bars.

* **Root Cause Analysis:**  
  The PyTorch `DataLoader` was configured with `num_workers=2`. Under restricted user slice resource limits on the multi-tenant Linux server, worker process spawning (`fork`) and inter-process communication shared memory buffers deadlocked.

* **Resolution:**  
  Modified `src/train.py` to pass `num_workers=0` to `get_dataloaders()`:
  ```python
  train_loader, val_loader = get_dataloaders(
      data_dir=config["data"]["data_dir"],
      batch_size=config["training"]["batch_size"],
      num_workers=0,
  )
  ```
  **Result:** Data loading runs synchronously on the main thread, avoiding multiprocessing deadlocks and resuming smooth training iteration.

---

## 3. Training Container OOM Kill (Exit Code 137)

* **Symptom / Problem:**  
  After applying `num_workers=0` to the explicit call-site in `train.py`, the container still exited with **exit code 137** (SIGKILL) immediately after printing `Starting epoch 1/10`. This is the Linux kernel's Out-of-Memory (OOM) killer signal.

* **Root Cause Analysis:**  
  `dataset.py`'s `get_dataloaders()` function had a **default parameter** of `num_workers=2`, and `train.py` never passed an explicit override. Docker Compose only executes the code as written in the image — so despite the intent, the DataLoader still spawned 2 worker processes at runtime. Each worker process independently mapped the full CIFAR-10 dataset into memory, tripling peak RAM usage (~375 MB × 3 ≈ 1.1 GB) plus model weights and gradients, exceeding available container memory and triggering the OOM kill.

* **Resolution:**  
  Changed the **default value** in `src/dataset.py` from `num_workers=2` to `num_workers=0`:
  ```python
  def get_dataloaders(
      data_dir: str,
      batch_size: int = 64,
      num_workers: int = 0,   # was 2 — caused OOM kill in containers
  ) -> tuple[DataLoader, DataLoader]:
  ```
  **Result:** Memory usage dropped to **~375 MiB** even with batch_size=64 on 4 GiB servers. Training completes all 10 epochs successfully.

---

## 4. Serving Startup Race Condition (`FileNotFoundError`)

* **Symptom / Problem:**  
  Starting serving containers before or alongside training threw `FileNotFoundError: [Errno 2] No such file or directory: '/app/checkpoints/classifier_v1.pt'` causing the container to crash loop (`CrashLoopBackOff`).

* **Root Cause Analysis:**  
  The inference server (`src/serve.py`) attempted to load model weights immediately upon application startup before the training job finished generating and saving the checkpoint file.

* **Resolution:**  
  Implemented a non-blocking retry wait loop in `src/serve.py` at application startup:
  ```python
  while not os.path.exists(checkpoint_path):
      logger.info(f"Waiting for checkpoint at {checkpoint_path}...")
      time.sleep(5)
  ```
  **Result:** The serving container gracefully waits until the training container outputs the model checkpoint file before loading the model into memory.

---

## 5. Kubernetes Checkpoint Persistence across Workloads

* **Symptom / Problem:**  
  In the initial Kubernetes design, model checkpoints created by the training batch `Job` were not visible to the serving `Deployment`.

* **Root Cause Analysis:**  
  Using ephemeral `emptyDir` volumes ties storage lifecycle strictly to individual pod lifetimes. Once the training job pod finished, its `emptyDir` was discarded.

* **Resolution:**  
  Created a dedicated `PersistentVolumeClaim` (`k8s/pvc.yaml`) named `checkpoint-pvc` with `ReadWriteOnce` access mode. Updated both `k8s/training-job.yaml` and `k8s/serving-deploy.yaml` to mount `checkpoint-pvc` at `/app/checkpoints`.  
  **Result:** Trained model weights persist across pod executions and are shared seamlessly from training job to serving pods.

---

## 6. Non-Standard Port Binding & Horizontal Pod Autoscaler (HPA) Constraints

* **Symptom / Problem:**  
  Requirements specified aligning container ports and establishing container autoscaling. Initial K8s deployment manifests lacked explicit container resource declarations.

* **Root Cause Analysis:**  
  Kubernetes HPA requires explicit resource `requests` (CPU and Memory) on target containers to compute utilization metrics; missing specs cause HPA to remain in `Unknown` state.

* **Resolution:**  
  - Updated `serving-deployment.yaml` containerPort and `serving-service.yaml` targetPort to standard port **`8080`** and service port **`80`**.
  - Added container resource `requests` (CPU 500m, RAM 1Gi) and `limits` (CPU 1000m, RAM 2Gi) in deployment manifests.
  - Configured `k8s/hpa.yaml` with an 80% CPU target utilization limit (min 1, max 3 replicas).

---

## 7. CIFAR-10 Upstream Server Throttling & Mirror Engine

* **Symptom / Problem:**  
  During containerized training on Kubernetes, the dataset download step stalled for over **42 minutes** (`100% [42:54, 66 KB/s]`), severely bottlenecking pipeline initiation.

* **Root Cause Analysis:**  
  The default `torchvision.datasets.CIFAR10` downloader points to a single university host (`cs.toronto.edu`) that aggressively rate-limits incoming traffic and lacks chunked streaming or mirror failovers.

* **Resolution:**  
  Implemented `ensure_cifar10_dataset()` in `src/dataset.py`:
  - Added multi-mirror failover supporting `cave.cs.toronto.edu` and cloud mirrors.
  - Added `CIFAR10_MIRROR_URL` environment variable support for private S3/GCS buckets.
  - Implemented 1 MB chunked streaming with live progress reporting and automatic `.tar.gz` extraction.
  - Enforced a persistent volume check that skips network calls if `/app/data/cifar-10-batches-py` exists.
  **Result:** Fast resilient downloads with immediate 0-second cache hits on persistent storage.

---

## 8. Ingress Subpath Routing & Path Rewriting

* **Symptom / Problem:**  
  Serving model predictions under a public subpath (`https://opssynergy.isroot.in/ml-training/*`) resulted in `404 Not Found` or `502 Bad Gateway` when requests reached the FastAPI backend.

* **Root Cause Analysis:**  
  FastAPI registers endpoints at `/health`, `/predict`, and `/docs`. Forwarding `/ml-training/health` without path rewriting caused FastAPI to receive the unstripped prefix `/ml-training/health`.

* **Resolution:**  
  Created `k8s/ingress.yaml` utilizing NGINX Ingress regex capture and rewrite annotations:
  ```yaml
  nginx.ingress.kubernetes.io/use-regex: "true"
  nginx.ingress.kubernetes.io/rewrite-target: /$2
  ```
  Path pattern `/ml-training(/|$)(.*)` correctly forwards requests to backend root routes (`/health`, `/predict`, `/docs`).
  **Result:** Clean HTTPS public ingress routing with automated Let's Encrypt certificates managed by cert-manager.

---

## 9. CI Test Discovery & Python Module Resolution

* **Symptom / Problem:**  
  Running `pytest tests/ -v` inside the GitHub Actions CI workflow threw `ModuleNotFoundError: No module named 'src'`.

* **Root Cause Analysis:**  
  Pytest does not automatically inject the repository root into `sys.path` during test module collection in isolated CI environments.

* **Resolution:**  
  - Created `pytest.ini` with `pythonpath = .` and `testpaths = tests`.
  - Updated `.github/workflows/ci.yml` to invoke tests via `python -m pytest tests/ -v`.
  **Result:** Seamless unit test discovery and 100% passing test execution in CI.

---

## Summary Table

| Issue Area | Initial Behavior | Root Cause | Implemented Solution | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Container Size** | 8.5 GB images on remote server | PyPI Linux wheels bundled ~7.5 GB CUDA libraries | Added PyTorch CPU index URL + `.dockerignore` | **1.4 GB image size** (~83% reduction) |
| **DataLoader Deadlock** | Stuck at Epoch 1 indefinitely | Multiprocessing `fork` deadlock with `num_workers=2` | Changed default to `num_workers=0` in `dataset.py` | Smooth batch progress |
| **OOM Kill (exit 137)** | Container killed after `Starting epoch 1/10` | `num_workers` default not overridden — workers tripled RAM usage | Fixed default in `get_dataloaders()` signature | **~375 MiB** peak memory, training completes |
| **Inference Server** | `FileNotFoundError` crash loop | Race condition before checkpoint output | Added retry/watch loop in `serve.py` | Reliable startup sequence |
| **K8s Storage** | Checkpoint lost after training | `emptyDir` volume lifecycle limits | Created shared `checkpoint-pvc` PVC | Shared persistent model storage |
| **K8s Autoscaling** | HPA inactive | Missing container resource requests/limits | Added CPU/Mem limits & port 8080 alignment | Active auto-scaling ready |
| **Dataset Download** | 42-minute download stall | Throttled single university server | Multi-mirror engine + chunked streaming | Resilient downloads + instant PVC cache |
| **Public Ingress** | 404/502 on subpath routing | FastAPI unstripped subpath mismatch | NGINX regex rewrite target (`/$2`) | Clean `opssynergy.isroot.in/ml-training/*` ingress |
| **CI Discovery** | `ModuleNotFoundError: src` | `sys.path` missing repo root in CI | Added `pytest.ini` with `pythonpath = .` | 100% CI automated test pass |
