# Porting to the Pi

```bash
.venv/bin/python cli.py train        # workstation
.venv/bin/python cli.py export       # -> out/models/*.onnx
scp -r out/models pi@station:~/aoi/out/    # includes the .onnx.data sidecars
# on the Pi:
python cli.py demo --backend onnx
```

Weights over ~1 MB export as an external `.onnx.data` file beside the `.onnx`.
Both must be copied together or the session loads with empty tensors.

`models.Backend` is the only seam. `TorchBackend` and `OnnxBackend` implement
the same two methods, so `pipeline.py` never learns which one it is talking to.
A `HailoBackend` slots in the same way.

Pi constraints carried over from the spec: USB3 camera rather than GigE
(~1 GB per board), keep 96×96 crops rather than full frames (~5 MB vs 1 GB per
board), accelerator on the single PCIe lane with storage on USB3, and training
stays on the workstation.

---

[← design notes index](README.md) · [repository README](../../README.md)
