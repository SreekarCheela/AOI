"""Small models plus the backend seam that lets training and deployment differ.

Training happens on the workstation in PyTorch; the station runs the exported
ONNX (and, on a Hailo-equipped Pi, the compiled graph). Both satisfy `Backend`,
so `pipeline` never learns which one it is talking to -- the port is a config
change, not a rewrite (spec section 7).

Anomaly detection is a per-package subspace model. For each package family, the
good patches from teach are pooled, average-pooled 4x, and reduced to a small
principal subspace. A patch is scored by how far it sits *outside* that subspace
-- the residual after projecting onto it. Training is one SVD; no epochs.

That works here because of an architectural consequence, not luck: patches
arrive CAD-cropped, de-rotated and scale-normalised, so good examples of one
package occupy a genuinely narrow subspace and anything malformed leaves it.

Two neural approaches were tried first and are recorded because the failures
are instructive:

  pixel autoencoder     separation x1.0  -- reconstruction error is dominated by
                                            background, identical either way, so
                                            it collapses to the mean image
  student-teacher       separation x1.4  -- a frozen *random* teacher is too easy
                                            a function; the student fits it
                                            everywhere, defects included. Real
                                            EfficientAD uses a pretrained teacher
                                            whose function is hard enough that
                                            matching it off-distribution fails
  per-package subspace  separation x32   <- shipped

Scoring globally rather than per package costs an order of magnitude on its own
(x1.1 vs x4-6 even for a crude distance-to-mean): a 0402 and a tantalum share no
appearance, so one model asked to call both normal calls almost anything normal.

The shipped model is also the most deployable of the three -- a mean vector and
a k x D matrix, about 150 kB per package, exporting to ONNX as plain matmuls
that run on any accelerator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import config

N_CH = len(config.CHANNELS)


def to_tensor(patches: np.ndarray) -> np.ndarray:
    """(N,H,W,C) uint8 -> (N,C,H,W) float32 in [0,1]."""
    a = np.asarray(patches, np.float32)
    if a.ndim == 3:
        a = a[None]
    if a.max() > 1.5:
        a = a / 255.0
    return np.ascontiguousarray(a.transpose(0, 3, 1, 2))


# ------------------------------------------------------------------ torch nets
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_classifier(n_classes: int = 2):
    torch, nn = _torch()

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))
            self.features = nn.Sequential(blk(N_CH, 16), blk(16, 32), blk(32, 64), blk(64, 64))
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            x = self.features(x)
            return self.head(x.mean(dim=(2, 3)))

    return TinyCNN()


POOL = 4                 # spatial decimation before the subspace fit
DEFAULT_K = 16           # subspace rank; x29 at k=8, x32 at k=16, x32 at k=32


def _pooled(x):
    """(N,C,H,W) -> (N, D) decimated feature vector, in torch channel-major order."""
    torch, nn = _torch()
    return nn.functional.avg_pool2d(x, POOL).flatten(1)


def fit_subspace(patches: np.ndarray, k: int = DEFAULT_K):
    """Fit the good-patch subspace for one package. Good examples only."""
    torch, _ = _torch()
    x = torch.from_numpy(to_tensor(patches))
    z = _pooled(x)
    mu = z.mean(dim=0)
    _, _, Vh = torch.linalg.svd(z - mu, full_matrices=False)
    return mu, Vh[: min(k, Vh.shape[0])].contiguous()


def build_subspace_anomaly(mu, V):
    """Score = distance from the good subspace. Pure linear algebra."""
    torch, nn = _torch()

    class SubspaceAnomaly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mu", mu)
            self.register_buffer("V", V)

        def forward(self, x):
            d = _pooled(x) - self.mu
            r = d - (d @ self.V.t()) @ self.V
            return torch.sqrt((r * r).sum(dim=1) + 1e-12)

    net = SubspaceAnomaly()
    net.eval()
    return net


# -------------------------------------------------------------------- backends
def _group(packages, n) -> dict[str, np.ndarray]:
    """Index patches by package so each goes to its own model."""
    if packages is None:
        return {}
    pk = np.asarray(packages)
    return {str(p): np.where(pk == p)[0] for p in np.unique(pk)}


def safe_name(package: str) -> str:
    """Package name -> a filename-safe key.

    Models are saved as `anomaly_{safe_name(pkg)}.pt` and both backends key
    their loaded dict by the filename stem, so every LOOKUP has to go through
    this too. It did not: `anomaly()` looked up by the raw package name, so any
    package whose name contained a slash or a space loaded under one key and was
    searched for under another, missed, and silently scored 0.0 -- a trained
    model contributing nothing, with no error. Latent only because every name in
    the hand-entered library happens to need no sanitising; the library is meant
    to be generated from datasheets, where `SOT-23-6 / TSOT` is ordinary.
    """
    return package.replace("/", "-").replace(" ", "_")


def check_name_collisions(packages: "Iterable[str]") -> None:
    """Refuse a package set whose names collide once sanitised.

    `safe_name` is not injective -- `A/B` and `A-B` both become `A-B` -- so two
    real packages can map to one filename and the second silently overwrites the
    first. Every component of the losing package would then be scored by the
    wrong model. Cheap to check, impossible to notice otherwise.
    """
    seen: dict[str, str] = {}
    for pkg in packages:
        key = safe_name(pkg)
        if key in seen and seen[key] != pkg:
            raise ValueError(
                f"package names {seen[key]!r} and {pkg!r} both sanitise to "
                f"{key!r}, so they would share one model file and the second "
                f"would overwrite the first")
        seen[key] = pkg


class Backend(ABC):
    """Everything `pipeline` is allowed to know about the models."""

    @abstractmethod
    def classify(self, patches: np.ndarray) -> np.ndarray:
        """(N,H,W,C) -> (N, n_classes) probabilities."""

    @abstractmethod
    def anomaly(self, patches: np.ndarray, packages: list[str] | None = None) -> np.ndarray:
        """(N,H,W,C) + per-patch package names -> (N,) anomaly score."""


class NullBackend(Backend):
    """Stand-in before anything is trained, so the rule layer still runs."""

    def classify(self, patches):
        n = len(np.atleast_1d(patches)) if patches.ndim == 4 else 1
        return np.tile(np.array([[1.0, 0.0]], np.float32), (n, 1))

    def anomaly(self, patches, packages=None):
        n = patches.shape[0] if patches.ndim == 4 else 1
        return np.zeros(n, np.float32)


class TorchBackend(Backend):
    def __init__(self, ckpt_dir: str | Path = "out/models") -> None:
        import torch
        self.torch = torch
        d = Path(ckpt_dir)
        self.clf = self.ae = None
        if (d / "classifier.pt").exists():
            sd = torch.load(d / "classifier.pt", map_location="cpu", weights_only=True)
            self.clf = build_classifier(sd["head.bias"].shape[0])
            self.clf.load_state_dict(sd)
            self.clf.eval()
        self.ae = {}
        for f in sorted(d.glob("anomaly_*.pt")):
            sd = torch.load(f, map_location="cpu", weights_only=True)
            self.ae[f.stem[len("anomaly_"):]] = build_subspace_anomaly(sd["mu"], sd["V"])

    def classify(self, patches):
        if self.clf is None:
            return NullBackend().classify(patches)
        x = self.torch.from_numpy(to_tensor(patches))
        with self.torch.no_grad():
            return self.torch.softmax(self.clf(x), dim=1).numpy()

    def anomaly(self, patches, packages=None):
        if not self.ae:
            return NullBackend().anomaly(patches)
        x = self.torch.from_numpy(to_tensor(patches))
        out = np.zeros(len(x), np.float32)
        for pkg, idx in _group(packages, len(x)).items():
            net = self.ae.get(safe_name(pkg))     # stored under the file stem
            if net is None:
                continue
            with self.torch.no_grad():
                out[idx] = net(x[idx]).numpy()
        return out


class OnnxBackend(Backend):
    """The deployment path -- what actually runs on the Pi."""

    def __init__(self, model_dir: str | Path = "out/models") -> None:
        import onnxruntime as ort
        d = Path(model_dir)
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 4          # Pi 5 has four cores
        self.clf = (ort.InferenceSession(str(d / "classifier.onnx"), opt,
                                         providers=["CPUExecutionProvider"])
                    if (d / "classifier.onnx").exists() else None)
        self.ae = {f.stem[len("anomaly_"):]:
                   ort.InferenceSession(str(f), opt, providers=["CPUExecutionProvider"])
                   for f in sorted(d.glob("anomaly_*.onnx"))}

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def classify(self, patches):
        if self.clf is None:
            return NullBackend().classify(patches)
        x = to_tensor(patches)
        z = self.clf.run(None, {self.clf.get_inputs()[0].name: x})[0]
        return self._softmax(z)

    def anomaly(self, patches, packages=None):
        if not self.ae:
            return NullBackend().anomaly(patches)
        x = to_tensor(patches)
        out = np.zeros(len(x), np.float32)
        for pkg, idx in _group(packages, len(x)).items():
            sess = self.ae.get(safe_name(pkg))    # stored under the file stem
            if sess is None:
                continue
            out[idx] = sess.run(None, {sess.get_inputs()[0].name: x[idx]})[0]
        return out


def load_backend(kind: str = "torch", model_dir: str | Path = "out/models") -> Backend:
    if kind == "onnx":
        return OnnxBackend(model_dir)
    if kind == "none":
        return NullBackend()
    return TorchBackend(model_dir)
