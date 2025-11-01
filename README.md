Hoa Viet Nam 2025 Flower Classifier
==================================

This repository contains a PyTorch transfer-learning pipeline for classifying six Vietnamese flower species (`Cuc`, `Dao`, `Lan`, `Mai`, `Sen`, `Tho`). The project is wired around configuration files so you can fine-tune quickly on the bundled dataset or plug in your own.

Project Layout
--------------

- `configs/default.yaml` – central experiment settings (data paths, augmentation, hyper-parameters).
- `src/` – training stack (`train.py`, `engine.py`, `model.py`, `data.py`, etc.).
- `scripts/prepare_split.py` – optional helper to split raw images into `train/` and `test/`.
- `tests/` – lightweight unit tests for preprocessing.
- `Datasets/HoaVietNam2025/HoaVietNam/` – expected dataset location (train/test subfolders already provided).
- `outputs/` – checkpoints written after training.
- `runs/` – TensorBoard event files (created automatically when enabled).

Prerequisites
-------------

1. Python 3.9+ (project tested on Windows with Python 3.9.12).
2. (Optional) GPU with CUDA for faster training; the pipeline automatically falls back to CPU.
3. PowerShell or another shell capable of running the shown commands.

Environment Setup
-----------------

```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dataset Check
-------------

Make sure the dataset is present at `Datasets/HoaVietNam2025/HoaVietNam` with the structure:

```
Datasets/HoaVietNam2025/HoaVietNam/
  train/
    Cuc/ *.jpg|png
    ...
  test/
    Cuc/ *.jpg|png
    ...
```

If you only have raw folders and need to create the split:

```powershell
python scripts/prepare_split.py --source path\to\raw_flowers --dest Datasets\HoaVietNam2025\HoaVietNam --test-size 0.2
```

Training
--------

Run a training session using the default configuration:

```powershell
python -m src.train --config configs/default.yaml --epochs 100
```

Useful overrides (each can be combined as needed):

- Change data path: `--data-root Datasets/HoaVietNam2025/HoaVietNam`
- Adjust batch size: `--batch-size 32`
- Switch backbone: `--backbone efficientnet_v2_s`
- Resume from checkpoint: `--resume outputs\hoavietnam2025_efficientnet_b0.pt`

The best-performing weights (based on validation accuracy) are saved to `outputs/<experiment_name>.pt`.

Evaluation
----------

To evaluate a saved checkpoint on the test split and print classification metrics:

```powershell
python -m src.evaluate --config configs/default.yaml --checkpoint outputs\hoavietnam2025_efficientnet_b0.pt
```

Inference
---------

Predict labels for one or more images:

```powershell
python -m src.predict --config configs/default.yaml --checkpoint outputs\hoavietnam2025_efficientnet_b0.pt --images path\to\image1.jpg path\to\image2.jpg
```

The command prints the top class and probability for each image.

Testing
-------

Run unit tests to ensure transforms output the expected tensor shape:

```powershell
python -m pytest tests\test_transforms.py
```

Configuration Tips
------------------

- Edit `configs/default.yaml` to tweak augmentation, optimizer type, learning rate, dropout, or logging.
- Set `training.patience` > 0 to re-enable early stopping; leave it at `0` to train for all epochs.
- Enable TensorBoard (`logging.tensorboard: true`); launch with:

  ```powershell
  tensorboard --logdir runs
  ```

Troubleshooting
---------------

- *Installation slow or fails*: ensure you have a stable network connection when installing PyTorch wheels; if needed, download the correct CUDA/CPU wheel manually.
- *Training very slow*: consider reducing `batch_size`, lowering augmentation intensity, or moving to a GPU environment.
- *Out-of-memory on GPU*: lower the batch size or switch to a lighter backbone (`efficientnet_b0`).
- *Mismatch between config and checkpoint*: checkpoints store the config used during training; the evaluation/predict scripts merge this with any runtime overrides.

With the steps above you can reproduce the baseline model, extend it with new data, and export predictions. Happy training!
