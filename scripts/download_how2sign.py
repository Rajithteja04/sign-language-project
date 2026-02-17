import argparse
import tarfile
from pathlib import Path

try:
    import gdown
except Exception as e:
    raise SystemExit("gdown is required. Install with: pip install gdown") from e

FILES = {
    "train": {
        "keypoints": "1TBX7hLraMiiLucknM1mhblNVomO9-Y0r",
        "translation": "1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f",
    },
    "val": {
        "keypoints": "1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w",
        "translation": "1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3",
    },
    "test": {
        "keypoints": "1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp",
        "translation": "1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG",
    },
}

def download_file(file_id: str, out_path: Path):
    if out_path.exists():
        print(f"Skipping existing: {out_path}")
        return out_path
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {out_path.name} ...")
    gdown.download(url, str(out_path), quiet=False)
    return out_path

def extract_tar(tar_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path.name} -> {dest}")
    with tarfile.open(tar_path, mode="r:*") as tf:
        tf.extractall(dest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="How2Sign")
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--translations", action="store_true")
    args = parser.parse_args()

    if not args.keypoints and not args.translations:
        parser.error("Select at least one: --keypoints or --translations")

    root = Path(args.root)
    cwd = Path.cwd()

    for split, ids in FILES.items():
        if args.keypoints:
            tar_name = f"{split}_2D_keypoints.tar.gz"
            tar_path = download_file(ids["keypoints"], cwd / tar_name)
            dest = root / "sentence_level" / split / "rgb_front" / "features"
            extract_tar(tar_path, dest)

        if args.translations:
            csv_name = f"how2sign_{split}.csv"
            csv_path = download_file(ids["translation"], cwd / csv_name)
            dest = root / "sentence_level" / split / "text" / "en" / "raw_text"
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / csv_name
            if not target.exists():
                csv_path.replace(target)
                print(f"Moved {csv_name} -> {target}")
            else:
                print(f"Already exists: {target}")

    print("Done.")

if __name__ == "__main__":
    main()
