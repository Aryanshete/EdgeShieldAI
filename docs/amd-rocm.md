# AMD / ROCm runtime

## Current local status

The Phase 1 development machine is a Windows device with an NVIDIA GeForce RTX
3050 Laptop GPU and a CPU-only PyTorch build (`torch.version.hip` is empty).
It cannot provide a genuine ROCm result. EdgeShield therefore uses CPU locally
and reports `backend: CPU`, never an AMD/ROCm claim.

`src/runtime.py` determines the backend from PyTorch runtime facts:

- `ROCm` requires both `torch.cuda.is_available()` and a non-empty
  `torch.version.hip` value.
- `CUDA` identifies a non-ROCm PyTorch accelerator; it is not presented as AMD.
- `CPU` is the fallback when no PyTorch accelerator is active.

Run this check on every target environment:

```bash
python scripts/verify_rocm.py
```

It exits successfully only when AMD ROCm is actually active. It also prints the
PyTorch, HIP, device, and backend facts that should be captured for the final
demo and performance report.

## Target AMD deployment

Deploy the repository to an AMD GPU instance (for example, an AMD Developer
Cloud allocation) that is supported by the selected ROCm release. Use AMD's
current [PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
and compatibility matrix for the exact GPU, OS, Python, driver, and ROCm
combination; those selections must be made for the actual instance.

For a Linux container deployment, `Dockerfile.rocm` starts from AMD's ROCm
PyTorch image and installs only the framework-independent application packages.
Build and run it on the AMD host with GPU devices exposed to the container:

```bash
docker build -f Dockerfile.rocm -t edgeshield-rocm .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video edgeshield-rocm
```

The container's default command runs `scripts/verify_rocm.py`. Only after that
check passes should model inference be run with the `cuda:0` device and the UI
show AMD ROCm as active.

## Windows note

AMD currently documents Windows PyTorch support for a limited set of supported
AMD Radeon and Ryzen hardware. It is not applicable to an NVIDIA GPU. Confirm
the exact hardware and driver against AMD's
[Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
before using a Windows target.
