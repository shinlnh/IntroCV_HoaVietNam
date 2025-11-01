import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def copy_subset(files, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, destination / src.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to raw dataset with class folders")
    parser.add_argument("--dest", required=True, help="Destination split root")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    dest_train = Path(args.dest) / "train"
    dest_test = Path(args.dest) / "test"

    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        if not files:
            continue
        train_files, test_files = train_test_split(
            files,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
        )
        copy_subset(train_files, dest_train / class_dir.name)
        copy_subset(test_files, dest_test / class_dir.name)
    print("Dataset split finished.")


if __name__ == "__main__":
    main()
