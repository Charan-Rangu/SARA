"""Create clean submission package and ZIP archive for SARA competition submission."""

import os
import sys
import shutil
import zipfile

def package_submission():
    root_dir = os.path.abspath(".")
    pkg_dir = os.path.join(root_dir, "submission_package")
    dist_dir = os.path.join(root_dir, "dist")
    zip_path = os.path.join(dist_dir, "SARA_competition_submission.zip")

    print("=== Creating SARA Competition Submission Package ===")

    # Clean existing dirs if present
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    os.makedirs(pkg_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    # 1. Essential root files
    root_files = [
        "README.md",
        "requirements.txt",
        "requirements-core.txt",
        "requirements-geo.txt",
        "requirements-ml.txt",
        ".env.example",
        ".gitignore"
    ]
    for rf in root_files:
        src = os.path.join(root_dir, rf)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pkg_dir, rf))
            print(f"  [COPIED] {rf}")

    # 2. Source directories to copy (excluding caches and oversized archives)
    dirs_to_copy = [
        "backend",
        "geospatial",
        "ml",
        "validation",
        "tests",
        "config",
        "docs"
    ]

    def ignore_patterns(path, names):
        ignored = set()
        for name in names:
            # Exclude pycache, temporary files, and .env
            if name == "__pycache__" or name.endswith(".pyc") or name.endswith(".pyo"):
                ignored.add(name)
            if name.startswith(".env") and name != ".env.example":
                ignored.add(name)
            if name.endswith(".tmp") or name.endswith(".log"):
                ignored.add(name)
            # Exclude large .tif files in docs or tests unless explicit small fixtures
            if name.endswith(".tif") and os.path.getsize(os.path.join(path, name)) > 10_000_000:
                ignored.add(name)
        return ignored

    for d in dirs_to_copy:
        src_d = os.path.join(root_dir, d)
        dst_d = os.path.join(pkg_dir, d)
        if os.path.exists(src_d):
            shutil.copytree(src_d, dst_d, ignore=ignore_patterns)
            print(f"  [COPIED DIR] {d}/")

    # 3. Copy essential lightweight sample fixtures for demo
    sample_subdirs = [
        os.path.join("data", "ml", "sen1floods11", "images"),
        os.path.join("data", "ml", "sen1floods11", "labels"),
        os.path.join("experiments", "unet")
    ]
    for sub in sample_subdirs:
        dst_sub = os.path.join(pkg_dir, sub)
        os.makedirs(dst_sub, exist_ok=True)

    # Copy Bolivia fixture if available
    bolivia_src = os.path.join(root_dir, "data", "ml", "sen1floods11", "images", "Bolivia_001_sar.tif")
    if os.path.exists(bolivia_src):
        shutil.copy2(bolivia_src, os.path.join(pkg_dir, "data", "ml", "sen1floods11", "images", "Bolivia_001_sar.tif"))
        print("  [COPIED FIXTURE] data/ml/sen1floods11/images/Bolivia_001_sar.tif")

    # Copy model checkpoint if available (< 10MB)
    ckpt_src = os.path.join(root_dir, "experiments", "unet", "best_model.pt")
    if os.path.exists(ckpt_src):
        shutil.copy2(ckpt_src, os.path.join(pkg_dir, "experiments", "unet", "best_model.pt"))
        print("  [COPIED MODEL] experiments/unet/best_model.pt (1.93M params)")

    # 4. Create ZIP archive
    print("\n--- Compressing to dist/SARA_competition_submission.zip ---")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(pkg_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, pkg_dir)
                zipf.write(file_path, arcname)
                total_files += 1
                total_uncompressed_bytes += os.path.getsize(file_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  Archive created: {zip_path}")
    print(f"  Total Files in Archive: {total_files}")
    print(f"  Uncompressed Size: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"  Compressed ZIP Size: {zip_size_mb:.2f} MB")

    # 5. Verify ZIP integrity
    print("\n--- Verifying ZIP Archive Integrity ---")
    with zipfile.ZipFile(zip_path, "r") as test_zip:
        corrupted = test_zip.testzip()
        if corrupted is None:
            print("  [PASS] ZIP archive is valid and uncorrupted.")
        else:
            print(f"  [FAIL] ZIP archive corrupted at: {corrupted}")
            
        file_list = test_zip.namelist()
        required_entries = [
            "README.md",
            "requirements.txt",
            ".env.example",
            "backend/main.py",
            "geospatial/change_detection.py",
            "geospatial/terrain_analysis.py",
            "validation/raster_validator.py",
            "docs/project_technical_summary.md",
            "docs/demo_script.md",
            "docs/final_submission_audit.md",
            "docs/submission_packaging_checklist.md"
        ]
        missing = [req for req in required_entries if req not in file_list]
        if not missing:
            print("  [PASS] All required root, source, and documentation files present in ZIP.")
        else:
            print(f"  [FAIL] Missing required entries in ZIP: {missing}")

if __name__ == "__main__":
    package_submission()
