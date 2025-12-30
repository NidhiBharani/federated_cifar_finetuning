from cifar10_pt_fl import Net

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "cifar10_pt_fl.py"

    # Create BaseFedJob with initial model
    job = BaseFedJob(
      name="cifar10_pt_fedavg",
      initial_model=Net(),
    )

    # Define the controller and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Add clients
    for i in range(n_clients):
        runner = ScriptRunner(script=train_script)
        job.to(runner, f"site-{i}")

    job.export_job("/tmp/nvflare/jobs/job_config")
    #job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")