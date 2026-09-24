#!/usr/bin/env bash
# ==============================================================================
# EdgeShield AI — Automated AMD Cloud & Docker Deployment Script
# Deploys EdgeShield AI on AMD Developer Cloud or ROCm Server
# ==============================================================================

set -euo pipefail

IMAGE_NAME="edgeshield-rocm:latest"

echo "================================================================="
echo "EDGESHIELD AI — AMD DEPLOYMENT AUTOMATION"
echo "================================================================="

# Step 1: Export ONNX model if not already present
if [ ! -f "models/yolov8n.onnx" ]; then
    echo "[*] Exporting YOLOv8 model for AMD ONNX Runtime..."
    python3 scripts/export_amd_model.py
else
    echo "[+] models/yolov8n.onnx already exists."
fi

# Step 2: Build AMD ROCm Docker Container
echo "[*] Building AMD ROCm container image '${IMAGE_NAME}'..."
docker build -f Dockerfile.rocm -t "${IMAGE_NAME}" .

# Step 3: Run Hardware Verification in Container
echo "[*] Running in-container ROCm hardware probe..."
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --group-add render \
    --ipc=host \
    "${IMAGE_NAME}" \
    python3 scripts/verify_rocm.py

# Step 4: Launch Service with Docker Compose
echo "[*] Starting EdgeShield AI SOC Dashboard on port 8501..."
docker compose -f docker-compose.rocm.yml up -d

echo "================================================================="
echo "[+] EdgeShield AI successfully deployed on AMD ROCm infrastructure!"
echo "[+] Access Operational SOC Dashboard: http://localhost:8501"
echo "================================================================="
