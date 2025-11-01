from src.data import build_transforms
from PIL import Image


def test_transforms_output_shape():
    train_tfms, val_tfms = build_transforms(224, {"rotation": 20})
    dummy = Image.new("RGB", (256, 256))
    out_train = train_tfms(dummy)
    out_val = val_tfms(dummy)
    assert out_train.shape == (3, 224, 224)
    assert out_val.shape == (3, 224, 224)
