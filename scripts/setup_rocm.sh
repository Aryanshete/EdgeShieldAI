#!/usr/bin/env bash
# ==============================================================================
# EdgeShield AI — AMD ROCm Host Initialization & Driver Verification Script
# Tested on: Ubuntu 22.04 / 24.04 LTS with AMD ROCm 6.x
# Target Hardware: AMD Instinct™ MI300/MI250, AMD Radeon™ RX 7000/6000
# ==============================================================================

set -euo pipefail

echo "================================================================="
echo "EDGESHIELD AI — AMD ROCm HOST INITIALIZATION"
echo "================================================================="

# 1. Check for AMD GPU PCIe devices
echo "[*] Checking for AMD GPU devices on PCIe bus..."
if lspci | grep -iE 'amd|radeon|advanced micro devices' > /dev/null; then
    echo "[+] Found AMD device(s):"
    lspci | grep -iE 'amd|radeon|advanced micro devices'
else
    echo "[!] Warning: No AMD PCIe device detected via lspci."
fi

# 2. Check /dev/kfd and /dev/dri permissions
echo "[*] Checking AMD kernel device nodes (/dev/kfd, /dev/dri)..."
if [ -e "/dev/kfd" ]; then
    echo "[+] /dev/kfd is present."
else
    echo "[-] /dev/kfd missing. Ensure AMD GPU kernel driver (amdgpu-dkms) is installed."
fi

if [ -d "/dev/dri" ]; then
    echo "[+] /dev/dri is present."
else
    echo "[-] /dev/dri missing."
fi

# Ensure user belongs to render and video groups
CURRENT_USER=$(whoami)
echo "[*] Adding user '${CURRENT_USER}' to 'video' and 'render' groups..."
sudo usermod -a -G video,render "$CURRENT_USER" || true

# 3. Check for rocm-smi
echo "[*] Checking rocm-smi utility..."
if command -v rocm-smi &> /dev/null; then
    echo "[+] rocm-smi is installed. Querying GPU status:"
    rocm-smi || true
else
    echo "[!] rocm-smi not found in PATH."
fi

# 4. Verify Python & PyTorch ROCm
echo "[*] Verifying PyTorch ROCm environment..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA / ROCm available:', torch.cuda.is_available())
print('HIP version:', getattr(torch.version, 'hip', None))
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
" || {
    echo "[-] PyTorch check failed. Installing recommended ROCm PyTorch wheel..."
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2
}

# 5. Run EdgeShield AMD verification probe
echo "[*] Running EdgeShield ROCm verification probe..."
python3 scripts/verify_rocm.py || true

echo "================================================================="
echo "[+] AMD ROCm host setup verification finished."
echo "================================================================="
