# NVFlare CIFAR-10 (PyTorch)

A minimal NVFlare federated learning example using a CNN on CIFAR-10. It includes a client training script and a job configuration that runs the NVFlare simulator.

## Project Files
- `cifar10_pt_fl.py`: model definition (`Net`), training loop, and evaluation.
- `job.py`: NVFlare job configuration and simulator run.
- `requirements.txt`: pinned Python dependencies.

## Quickstart
```bash
python -m venv nvflare-cifar-env
source nvflare-cifar-env/bin/activate
pip install -r requirements.txt
```

Run the NVFlare simulator:
```bash
python job.py
```

Run the client training script directly:
```bash
python cifar10_pt_fl.py
```

## Notes
- CIFAR-10 is downloaded to `/tmp/nvflare/data`.
- Simulator output is written to `/tmp/nvflare/jobs/workdir`.
- `job.py` requests GPU `"0"` when running the simulator; the training script auto-detects CUDA.

## Requirements
Python 3.10+ recommended. GPU is optional but speeds up training.
