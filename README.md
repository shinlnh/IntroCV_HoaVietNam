# Hoa Viet Nam 2025 Classifier

## Quickstart
```
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

python src/train.py --config configs/default.yaml

python src/evaluate.py --checkpoint outputs/hoavietnam2025_efficientnet_b0.pt

python src/predict.py --checkpoint outputs/hoavietnam2025_efficientnet_b0.pt --images path/to/image.jpg
```

- C?p nh?t `configs/default.yaml` ?? ch?nh ???ng d?n d? li?u, backbone ho?c hyper-parameters.
- TensorBoard logs s? ???c l?u trong th? m?c `runs/`.
- `scripts/prepare_split.py` h? tr? t?o train/test n?u ch? c? raw images.
