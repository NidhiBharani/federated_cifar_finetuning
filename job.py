from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from cifar10_pt_fl import Net

N_CLIENTS = 2
NUM_ROUNDS = 2
TRAIN_SCRIPT = "cifar10_pt_fl.py"

if __name__ == "__main__":
    # Create BaseFedJob with initial model
    job = BaseFedJob(
        name="cifar10_pt_fedavg",
        initial_model=Net(),
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=N_CLIENTS,
        num_rounds=NUM_ROUNDS,
    )
    job.to_server(controller)

    # Add clients
    for i in range(N_CLIENTS):
        runner = ScriptRunner(script=TRAIN_SCRIPT)
        job.to(runner, f"site-{i}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    # job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
