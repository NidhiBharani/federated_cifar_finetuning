# Federated CIFAR-10 Fine-Tuning with NVFlare and PyTorch

This repository is a compact federated learning example built around NVFlare, PyTorch, and CIFAR-10. It defines:

- an NVFlare job configuration in `job.py`
- a PyTorch client training script in `cifar10_pt_fl.py`
- a small CNN (`Net`) used as the shared model across all federated rounds

The codebase is intentionally small, but there are a few runtime details that matter:

- `job.py` currently exports an NVFlare job to `/tmp/nvflare/jobs/job_config`
- the simulator call in `job.py` is present but commented out
- `cifar10_pt_fl.py` is written as an NVFlare site script and calls `nvflare.client.init()`, so it should be treated primarily as a federated client entrypoint rather than a plain standalone trainer

## Documentation Map

- [Project Overview](docs/project-overview.md)
- [Runtime Flow](docs/runtime-flow.md)
- [Code Reference](docs/code-reference.md)
- [Development Guide](docs/development-guide.md)

## Quick Start

Create a local environment and install the pinned dependencies:

```bash
python -m venv nvflare-cifar-env
source nvflare-cifar-env/bin/activate
pip install -r requirements.txt
```

Export the NVFlare job configuration:

```bash
python job.py
```

This writes the generated job package under `/tmp/nvflare/jobs/job_config`.

## Current Configuration Snapshot

- Clients: `2`
- Federated rounds: `2`
- Local epochs per round: `2`
- Batch size: `4`
- Dataset location: `/tmp/nvflare/data`
- Saved local model checkpoint: `./cifar_net.pth`
- Runtime device: auto-detected from CUDA availability

## Repository Layout

```text
.
|-- README.md
|-- docs/
|   |-- project-overview.md
|   |-- runtime-flow.md
|   |-- code-reference.md
|   `-- development-guide.md
|-- cifar10_pt_fl.py
|-- job.py
`-- requirements.txt
```

## Important Notes

- The top-level README that originally shipped with the repository described `python job.py` as running the simulator directly. In the current code, that is no longer true because `job.simulator_run(...)` is commented out.
- The client script reports an `accuracy` metric based on the received model weights, while returning locally trained weights for aggregation. That behavior is documented in [Code Reference](docs/code-reference.md).
- CIFAR-10 data and exported NVFlare artifacts are written under `/tmp`, so they are ephemeral across machine cleanup or reboot events.
