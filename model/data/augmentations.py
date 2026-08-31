"""Train/eval image transforms.

Augmentation choices target the failure modes that matter for solder-joint
imaging specifically: lighting/exposure variation (the eventual rig has its
own dedicated LED lighting, but pretraining data comes from different
capture setups) and minor camera angle/focus variation.
- Flips are safe: defect presence isn't orientation-dependent, so both
  horizontal and vertical flips are used freely.
- Rotation is kept small (+/-8 degrees) since large rotations could misalign
  or crop out small joints near the frame edge.
- Color jitter is kept mild deliberately: color is itself a defect cue (cold
  joints read duller/greyer than a good joint), so jitter strong enough to
  mask that would teach the model to ignore a real signal.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
