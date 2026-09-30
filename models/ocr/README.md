# OCR model files (silkscreen names)

`aoi/photo/names.py` reads part names with PP-OCR through RapidOCR on
onnxruntime. It needs four ONNX files in this folder, about 108 MB together:

| File | Used for |
|---|---|
| `PP-OCRv6_det_small.onnx` | text detector |
| `PP-OCRv6_rec_medium.onnx` | recogniser, board-wide and close-up reads |
| `PP-OCRv6_rec_small.onnx` | recogniser, the 1.5x close-up read |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | loaded by RapidOCR, never run |

Their SHA-256 hashes are pinned in `names.MODELS`. Nothing is downloaded while
the app runs: with a file missing, teach leaves parts unnamed and says which
file is missing.

The files are not in git. Install them once per machine:

```bash
# on a machine with network: download, check every hash, install here
python models/ocr/fetch.py

# on the offline machine: copy this folder over (USB etc.), then
python -m aoi.photo.names install-models --from /path/to/copied/ocr   # copies and checks
python -m aoi.photo.names verify-models                                # checks what is here
```

On a machine that already ran RapidOCR, `--from .venv/lib/python3.14/site-packages/rapidocr/models`
also works: the files are the same, and the hashes are checked either way.

`AOI_OCR_MODELS=/some/dir` points the app at another folder.

License: the PaddleOCR models are Apache-2.0 (PaddlePaddle/PaddleOCR). The ONNX
conversions are RapidOCR's (RapidAI/RapidOCR), also Apache-2.0.
