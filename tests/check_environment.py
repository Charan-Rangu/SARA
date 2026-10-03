import sys
import os

def check_environment():
    print("=" * 60)
    print("SARA ENVIRONMENT CHECK")
    print("=" * 60)
    
    executable = sys.executable
    print(f"Python Executable: {executable}")
    print(f"Python Version:    {sys.version.split()[0]}")
    
    # Verify .venv location
    is_project_venv = ".venv" in executable or "sara" in executable.lower()
    print(f"In Project .venv:  {is_project_venv}")
    print("-" * 60)

    packages = [
        "torch",
        "numpy",
        "scipy",
        "rasterio",
        "geopandas",
        "shapely",
        "pyproj",
        "networkx",
        "fastapi"
    ]

    missing = []
    print("PACKAGE VERSIONS:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed (no __version__)")
            print(f"  - {pkg:<12}: {version}")
        except ImportError as e:
            print(f"  - {pkg:<12}: MISSING ({e})")
            missing.append(pkg)

    print("-" * 60)
    # Check PyTorch CUDA status
    try:
        import torch
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device:           {torch.cuda.get_device_name(0)}")
            print(f"CUDA Runtime:         {torch.version.cuda}")
    except Exception as e:
        print(f"PyTorch Check Error:  {e}")

    print("=" * 60)
    if missing:
        print(f"STATUS: FAILED — Missing packages: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("STATUS: SUCCESS — All core dependencies imported cleanly!")

if __name__ == "__main__":
    check_environment()
