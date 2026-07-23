import gzip
import shutil
import tarfile
from pathlib import Path

ipk_dir = Path("data/raw/ipk_downloads/")
extract_dir = Path("data/raw/benign_mips/dir/")
binary_dir = Path("data/raw/benign_mips/executables")

extract_dir.mkdir(exist_ok=True)
binary_dir.mkdir(exist_ok=True)

for pkg in ipk_dir.glob("*.ipk"):
    print(f"Processing {pkg.name}")

    workdir = extract_dir / pkg.stem
    workdir.mkdir(parents=True, exist_ok=True)

    # Decompress gzip -> package.tar
    outer_tar = workdir / "package.tar"
    with gzip.open(pkg, "rb") as fin:
        with open(outer_tar, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    # Extract outer tar
    with tarfile.open(outer_tar, "r:") as tf:
        tf.extractall(workdir)

    # Find and extract data.tar.*
    data_tar = None
    for name in ("data.tar.gz", "data.tar.xz", "data.tar.bz2", "data.tar"):
        p = workdir / name
        if p.exists():
            data_tar = p
            break

    if data_tar is None:
        print(f"  No data.tar found in {pkg.name}")
        continue

    with tarfile.open(data_tar, "r:*") as tf:
        tf.extractall(workdir)

    # Copy only ELF files
    for f in workdir.rglob("*"):
        if not f.is_file():
            continue

        try:
            with open(f, "rb") as fp:
                if fp.read(4) != b"\x7fELF":
                    continue

            dest = binary_dir / f.name

            if dest.exists():
                dest = binary_dir / f"{pkg.stem}_{f.name}"

            shutil.copy2(f, dest)
            print(f"  Copied {dest.name}")

        except Exception:
            pass

extract_dir.rmdir()
print("Done!")
