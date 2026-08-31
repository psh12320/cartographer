# GPU Embedding Handoff

This task generates Cartographer's 50,000 normalized BGE product embeddings. The output is portable across machines and is rejected automatically if it was built from a different catalog or product ordering.

## One-command Windows path

From a fresh checkout on an NVIDIA laptop, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_embeddings_gpu.ps1 -CreateArchive
```

The script creates `.venv-embed`, installs a CUDA PyTorch wheel and the declared dependencies, downloads and checksums the official catalog if it is missing, verifies CUDA, retries batch sizes down to 32, validates the final matrix, and creates `cartographer-bge-artifacts.zip`. If the default CUDA 12.6 wheel is unsuitable for the installed driver, pass the wheel index chosen at <https://pytorch.org/get-started/locally/> using `-TorchIndexUrl`.

## 1. Check out the checkpoint and obtain the catalog

Use the latest shared branch containing `cartographer/build_embeddings.py`. Do not modify or reorder `data/catalog.jsonl`.

Verify that the catalog contains 50,000 rows and matches the organizer checksum:

```powershell
(Get-Content -LiteralPath data\catalog.jsonl | Measure-Object -Line).Lines
(Get-FileHash -Algorithm SHA256 data\catalog.jsonl).Hash
```

The expected decompressed `catalog.jsonl` SHA256 is `DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67`. The release archive `catalog.jsonl.gz` has SHA256 `07FD142631FD6B03E2B4D09988C3EB7D53720E9D57010C79DB48EEAADA50A8F8`.

## 2. Create a GPU environment

Python 3.10 or 3.11 is recommended. In PowerShell:

```powershell
py -3.11 -m venv .venv-embed
.\.venv-embed\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build using the command recommended by the official PyTorch installer for the laptop's NVIDIA driver. Then install the remaining dependencies:

```powershell
python -m pip install numpy sentence-transformers huggingface_hub
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Do not continue unless the output says `cuda True`.

## 3. Download and verify BGE

The model is public and does not require a Hugging Face token:

```powershell
hf download BAAI/bge-small-en-v1.5 --local-dir data/cartographer_index/bge-small-en-v1.5 --exclude "*.onnx" "*.bin" "*.tflite" "*.msgpack" "openvino/*"
hf cache verify BAAI/bge-small-en-v1.5 --local-dir data/cartographer_index/bge-small-en-v1.5
```

The `hf` command is the current CLI; do not use the deprecated `huggingface-cli` command.

## 4. Generate the matrix

Start with batch size 128:

```powershell
python -m cartographer.build_embeddings --device cuda --batch-size 128 --dtype float32
```

If CUDA reports an out-of-memory error, retry with `--batch-size 64`, then `32`. The command writes files only after encoding completes and immediately verifies their checksums and row alignment.

Successful output must report:

- `verified: true`
- `rows: 50000`
- `dimensions: 384`
- `dtype: float32`

## 5. Re-verify and transfer

```powershell
python -m cartographer.build_embeddings --verify-only
Compress-Archive -Path data\cartographer_index\embeddings.npy,data\cartographer_index\embeddings_manifest.json,data\cartographer_index\bge-small-en-v1.5 -DestinationPath cartographer-bge-artifacts.zip -Force
Get-FileHash -Algorithm SHA256 cartographer-bge-artifacts.zip
```

Send both `cartographer-bge-artifacts.zip` and the printed ZIP SHA256. Do not commit the ZIP, embedding matrix, model files, catalog, credentials, or Hugging Face cache to Git.

After extraction on another machine, place the three artifacts under `data/cartographer_index/` and run:

```powershell
python -m cartographer.build_embeddings --verify-only
```

The receiver should not use the embeddings unless this final verification succeeds.
