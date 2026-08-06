# Dockerfile for the MLOps pipeline (single image)
# ---------------------------------------------------
# NOTE: The build expects a `configs/` directory at the repository root
# containing the YAML configuration files (e.g., training_config.yaml,
# serving_config.yaml). If the directory is missing, the build will fail
# with a "not found" error. Ensure you add the required config files before
# building or create an empty placeholder directory.

FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy source code
COPY src/ ./src/
# Copy configuration files into the container's /app directory
COPY configs/ ./configs/
# Default command runs training; Docker‑Compose can override for serving
ENTRYPOINT ["python", "-m", "src.train"]
