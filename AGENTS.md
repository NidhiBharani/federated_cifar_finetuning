# Repository Guidelines

## Project Structure & Module Organization
This repository is a small NVFlare + PyTorch example focused on CIFAR-10. Key files live at the repo root:
- `cifar10_pt_fl.py`: client-side training script, model definition (`Net`), and local evaluation.
- `job.py`: NVFlare job configuration and simulator run entrypoint.
- `requirements.txt`: pinned Python dependencies.
- `nvflare-cifar-env/`: local virtual environment directory (if used); `__pycache__/` holds runtime bytecode.

## Build, Test, and Development Commands
- `python -m venv nvflare-cifar-env` and `source nvflare-cifar-env/bin/activate`: create/activate the local virtualenv.
- `pip install -r requirements.txt`: install pinned dependencies.
- `python job.py`: run the NVFlare federated learning simulator (writes to `/tmp/nvflare/jobs/workdir`, uses `gpu="0"`).
- `python cifar10_pt_fl.py`: run the client training loop directly; downloads CIFAR-10 to `/tmp/nvflare/data`.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, `snake_case` for variables/functions, `CapWords` for classes.
- Keep configuration constants near the top of modules (e.g., `DATASET_PATH`, `DEVICE`).
- No formatter or linter is configured; keep diffs small and readable.

## Testing Guidelines
- There is no test framework or test suite in this repo.
- If adding tests, prefer `pytest`, place files under `tests/`, and name them `test_*.py`.

## Commit & Pull Request Guidelines
- This directory does not include Git history, so there is no established commit convention to follow.
- Use short, imperative commit subjects (e.g., “Add CIFAR-10 metrics logging”), with a brief body when context helps.
- PRs should describe the change, how to run it locally, and any data/compute requirements (GPU, dataset paths).

## Security & Configuration Tips
- Training downloads CIFAR-10 into `/tmp/nvflare/data`; simulator artifacts go to `/tmp/nvflare/jobs`. Ensure local permissions and clean up large artifacts when needed.
- GPU usage is controlled in `job.py` (`gpu="0"`) and `cifar10_pt_fl.py` auto-detects CUDA.
