import sys
import torch

def verify_gpu():
    print("PyTorch version:", torch.__version__, flush=True)
    print("PyTorch CUDA runtime:", torch.version.cuda, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch could not access the GPU")

    device_name = torch.cuda.get_device_name(0)
    print("GPU:", device_name, flush=True)

    props = torch.cuda.get_device_properties(0)
    vram_gb = round(props.total_memory / (1024**3), 2)
    print("VRAM GB:", vram_gb, flush=True)

    # Tensor computation test on GPU
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b

    torch.cuda.synchronize()

    print("Matrix multiplication:", c.shape, flush=True)
    print("Result device:", c.device, flush=True)
    allocated_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
    print("Allocated memory MB:", allocated_mb, flush=True)

    assert c.is_cuda, "Result tensor is not on CUDA!"
    print("GPU TEST PASSED", flush=True)

if __name__ == "__main__":
    try:
        verify_gpu()
    except Exception as e:
        print(f"GPU Test Failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
