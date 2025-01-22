import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter


from dataset.dataset import RADIal

from dataset.encoder import ra_encoder

import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
#from utils.evaluation import run_evaluation
import torch.nn as nn
import matplotlib.pyplot as plt

import sys
import tempfile

import torch.distributed as dist
import socket
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import optuna
from optuna.trial import TrialState
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback


from torch.utils.data import Dataset, DataLoader

from model.FFTRadNet_noseg import FFTRadNet
from dataset.dataloader_ddp import CreateDataLoaders
from dataset.matlab_dataset_ddp import MATLAB
from utils.evaluation_ddp import run_evaluation
from multiprocessing import Process, Barrier


from functools import partial
import time

# Utility Functions

def get_master_addr():
    """
    Retrieve the IP address of the current machine to use as the DDP master address.
    Returns:
        str: The IP address of the current machine.
    """
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print("Master IP address: ", ip_address)
        return ip_address
    except socket.error as e:
        print(f"Failed to get IP address: {e}")
        raise


def find_free_port():
    """
    Find an available port on the current machine.
    Returns:
        int: An available port number.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            free_port = s.getsockname()[1]
            print('Free port: ', free_port)
            return free_port
    except socket.error as e:
        print(f"Failed to find a free port: {e}")
        raise

def ddp_setup(rank, world_size):
    """
    Sets up the Distributed Data Parallel (DDP) environment.
    Args:
        rank (int): Unique identifier for the process.
        world_size (int): Total number of processes in the distributed training.
    """
    try:
        master_addr = get_master_addr()
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = '38853'  # Preset port for A100 GPUs.

        # Initialize the process group with the NCCL backend.
        start_time = time.time()
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        end_time = time.time()
        print(f"DDP Initialization took {end_time - start_time:.2f} seconds.")
        torch.cuda.set_device(rank)
        print(f"Process {rank}/{world_size} initialized.")
    except Exception as e:
        print(f"Failed to set up DDP: {e}")
        raise

# Trainer Class

class Trainer:
    """
    Handles training and evaluation of the model in a DDP setup.
    """
    def __init__(self, model, train_data, val_data, optimizer, gpu_id, save_every, config, optuna_config, scheduler, encoder):
        self.gpu_id = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.config = config
        self.optuna_config = optuna_config
        self.scheduler = scheduler
        self.encoder = encoder

    def _compute_f1_score(self, TP, FP, FN):
        """
        Compute the F1 score along with precision and recall.
        Args:
            TP (float): True positives.
            FP (float): False positives.
            FN (float): False negatives.
        Returns:
            Tuple[float, float, float]: F1 score, precision, recall.
        """
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    def _aggregate_metrics(self, local_TP, local_FP, local_FN, device):
        """
        Aggregate metrics across multiple devices using DDP.
        Args:
            local_TP, local_FP, local_FN (float): Local metrics.
            device: Current device.
        Returns:
            Tuple[float, float, float]: Global aggregated metrics.
        """
        TP_tensor = torch.tensor([local_TP], dtype=torch.float).to(device)
        FP_tensor = torch.tensor([local_FP], dtype=torch.float).to(device)
        FN_tensor = torch.tensor([local_FN], dtype=torch.float).to(device)

        dist.all_reduce(TP_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(FP_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(FN_tensor, op=dist.ReduceOp.SUM)

        return TP_tensor.item(), FP_tensor.item(), FN_tensor.item()
   
    def _run_batch(self, source, targets):
        """
        Run a single training batch.
        Args:
            source: Input data.
            targets: Ground truth labels.
        Returns:
            float: Batch loss.
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        classif_loss, reg_loss = pixor_loss(output, targets, self.config['losses'])
        classif_loss *= self.config['losses']['weight'][0]
        reg_loss *= self.config['losses']['weight'][1]
        loss = classif_loss + reg_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch, total_train_loss, kbar):
        """
        Execute a single epoch of training.
        Args:
            epoch (int): Current epoch number.
            total_train_loss (float): Accumulated training loss.
            kbar: Progress bar object.
        Returns:
            float: Updated total training loss.
        """
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for i, data in enumerate(self.train_data):
            source = data[0].to(self.gpu_id).float()
            targets = data[1].to(self.gpu_id).float()
            loss = self._run_batch(source, targets)
            total_train_loss += loss
        self.scheduler.step()
        return total_train_loss

    def _run_train_evaluation(self, epoch):
        """
        Run training evaluation for a given epoch.

        This method evaluates the model on the training data, calculates local and global metrics,
        and computes F1 score, mean Average Precision (mAP), and mean Average Recall (mAR).

        Args:
            epoch (int): The current epoch number.

        Returns:
            tuple: F1_score, mAP, and mAR calculated for the training data.
        """
        self.model.eval()

        # Perform evaluation on the training data
        train = run_evaluation(self.model,
                                      self.train_data,
                                      self.encoder,
                                      self.gpu_id,
                                      self.optuna_config,
                                      check_perf=(epoch >= 2),
                                      detection_loss=pixor_loss,
                                      losses_params=self.config['losses'])

        # Extract local metrics
        local_TP = train['TP']
        local_FP = train['FP']
        local_FN = train['FN']
        print("self.gpu_id: ", self.gpu_id)
        print("local_TP: ", local_TP)
        print("local_FP: ", local_FP)
        print("local_FN: ", local_FN)

        # Aggregate metrics across GPUs and compute F1 score, mAP, and mAR
        global_TP, global_FP, global_FN = self._aggregate_metrics(local_TP, local_FP, local_FN)
        F1_score, mAP, mAR = self._compute_f1_score(global_TP, global_FP, global_FN)

        return F1_score, mAP, mAR

    def _run_evaluation(self, epoch, total_val_loss):
        """
        Run validation evaluation for a given epoch.

        This method evaluates the model on the validation data, calculates local and global metrics,
        and computes F1 score, mean Average Precision (mAP), and mean Average Recall (mAR). It also
        updates the total validation loss.

        Args:
            epoch (int): The current epoch number.
            total_val_loss (float): The cumulative validation loss up to the current epoch.

        Returns:
            tuple: F1_score, mAP, mAR, and updated total_val_loss.
        """
        self.model.eval()

        # Perform evaluation on the validation data
        eval = run_evaluation(self.model,
                                      self.val_data,
                                      self.encoder,
                                      self.gpu_id,
                                      self.optuna_config,
                                      check_perf=(epoch >= 2),
                                      detection_loss=pixor_loss,
                                      losses_params=self.config['losses'])

        # Extract local metrics
        local_TP = eval['TP']
        local_FP = eval['FP']
        local_FN = eval['FN']
        print("self.gpu_id: ", self.gpu_id)
        print("local_TP: ", local_TP)
        print("local_FP: ", local_FP)
        print("local_FN: ", local_FN)

        # Aggregate metrics across GPUs and compute F1 score, mAP, and mAR
        global_TP, global_FP, global_FN = self._aggregate_metrics(local_TP, local_FP, local_FN)
        F1_score, mAP, mAR = self._compute_f1_score(global_TP, global_FP, global_FN)

        # Print evaluation summary
        print(f"Epoch {epoch} | Evaluation Loss: {eval['loss']/ len(self.val_data.dataset):.4f} | mAP: {eval['mAP']:.4f} | mAR: {eval['mAR']:.4f} | F1 Score: {F1_score:.4f}")
        total_val_loss += eval['loss']

        return F1_score, mAP, mAR, total_val_loss

    def _save_checkpoint(self, epoch):
        """
        Save the model's state as a checkpoint.

        This method saves the model's state dictionary to a file named "checkpoint.pt". If the model
        is wrapped in a distributed data-parallel wrapper, it extracts the state dictionary of the
        underlying model.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """
        # Get the model state dictionary, handling distributed data-parallel models
        ckp = self.model.state_dict() if not hasattr(self.model, 'module') else self.model.module.state_dict()
        PATH = "checkpoint.pt"

        # Save the checkpoint
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, epoch: int, train_loss: int, val_loss: int, kbar):
    """
    Train the model for one epoch and perform evaluation.

    Args:
        epoch (int): The current training epoch.
        train_loss (int): Initial value for cumulative training loss.
        val_loss (int): Initial value for cumulative validation loss.
        kbar: Progress bar object for tracking the training process.

    Returns:
        tuple: A tuple containing the following:
            - f1 (float): F1 score calculated during evaluation.
            - mAP (float): Mean Average Precision (mAP) calculated during evaluation.
            - mAR (float): Mean Average Recall (mAR) calculated during evaluation.
            - total_val_loss (float): Total validation loss for the epoch.
            - total_train_loss (float): Total training loss for the epoch.
    """
    self.model.train()  # Set the model to training mode.

    # Run the training process for the current epoch.
    total_train_loss = self._run_epoch(epoch, train_loss, kbar)

    # Run the evaluation on the validation dataset.
    f1, mAP, mAR, total_val_loss = self._run_evaluation(epoch, val_loss)

    # Save the model checkpoint if it's the designated GPU and the save interval is met.
    if self.gpu_id == 0 and epoch % self.save_every == 0:
        self._save_checkpoint(epoch)

    # Return the evaluation metrics and losses.
    return f1, mAP, mAR, total_val_loss, total_train_loss


# ddp
def load_train_objs(config, optuna_config):
    """
    Load the training objects including the dataset, model, optimizer, and scheduler.

    Args:
        config (dict): Configuration dictionary containing dataset, model, and training parameters.
        optuna_config (dict): Configuration dictionary with hyperparameters optimized by Optuna.

    Returns:
        tuple: A tuple containing the dataset, model, optimizer, scheduler, and encoder.
    """
    print("======== loading dataset and model =======")

    # Initialize the encoder with geometry and statistics settings from the configuration.
    enc = ra_encoder(geometry=config['dataset']['geometry'], 
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)

    # Load the MATLAB dataset using the specified directory, statistics, and encoder.
    dataset = MATLAB(root_dir=config['dataset']['root_dir'], 
                     folder_dir=config['dataset']['data_folder'], 
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode)

    # Initialize the FFTRadNet model with the given configuration and Optuna-tuned parameters.
    model = FFTRadNet(blocks=config['model']['backbone_block'],
                      mimo_layer=optuna_config['model']['mimo_layer'],  # Optuna tuning
                      Ntx=config['model']['NbTxAntenna'],
                      Nrx=config['model']['NbRxAntenna'],
                      channels=config['model']['channels'], 
                      regression_layer=2, 
                      detection_head=config['model']['DetectionHead'])

    # Set up the optimizer with Optuna-tuned learning rate.
    lr = optuna_config['optimizer']['lr']  # Learning rate
    step_size = optuna_config['optimizer']['step_size']  # Step size for learning rate scheduler
    gamma = float(config['lr_scheduler']['gamma'])  # Decay rate for learning rate
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Set up the learning rate scheduler.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return dataset, model, optimizer, scheduler, enc

# ddp
def prepare_dataloader(config, dataset: Dataset, batch_size):
    """
    Prepare dataloaders for training and validation.

    Args:
        config (dict): Configuration dictionary containing dataloader parameters and random seed.
        dataset (Dataset): The dataset to be split into training and validation sets.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    print("======== preparing dataloader =======")

    # Create data loaders for training and validation sets.
    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])

    return train_loader, val_loader

def print_memory_summary():
    """
    Print a summary of the CUDA memory usage.

    This method provides a detailed summary of the current CUDA memory allocation and utilization.

    Returns:
        None
    """
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary())

##########
# optuna #
##########

def objective(single_trial, config, rank, world_size):
    """
    The objective function for Optuna optimization.

    This function defines the training and evaluation pipeline for the model. It uses Optuna to 
    tune hyperparameters and saves the best-performing configuration. The function also integrates
    distributed training and wandb for tracking experiments.

    Args:
        single_trial (optuna.Trial): The trial object provided by Optuna.
        config (dict): The main configuration dictionary containing dataset, model, and training settings.
        rank (int): The rank of the current process in distributed training.
        world_size (int): Total number of processes in distributed training.

    Returns:
        float: The final F1 score achieved by the model.
    """
    # Create an output directory for the current experiment
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

    # Save the configuration to a JSON file
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # Create a TorchDistributedTrial instance
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    # Define hyperparameter search space
    optuna_config = {
        "optimizer": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-4, log=True),
            "step_size": trial.suggest_categorical("step_size", [10]),
        },
        "model": {
            "mimo_layer": trial.suggest_categorical("mimo_layer", [192]),
        },
    }

    # Load dataset, model, optimizer, and scheduler
    dataset, model, optimizer, scheduler, encoder = load_train_objs(config, optuna_config)

    # Prepare data loaders
    train_data, val_data = prepare_dataloader(config, dataset, config['dataloader']['train']['batch_size'])

    # Initialize trainer
    trainer = Trainer(
        model, train_data, val_data, optimizer, rank,
        config['save_every'], config, optuna_config, scheduler, encoder
    )

    # Number of epochs
    num_epochs = 5

    # Initialize wandb for experiment tracking
    if rank == 0:
        wandb.init(
            project=config['optuna_project'],
            entity="chu06-imec",
            config=dict(trial.params),
            group="FFTRadNet_optimization",
            reinit=True,
        )

    history = {
        "train_loss": [], "val_loss": [], "lr": [], "mAP": [], 
        "mAR": [], "F1_score": [], "train_mAP": [], "train_mAR": [], "train_F1_score": []
    }

    for epoch in range(num_epochs):
        print(f"--- Epoch: {epoch} ---")

        # Train and evaluate the model for the current epoch
        kbar = pkbar.Kbar(target=len(train_data), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        F1_score, mAP, mAR, val_loss, train_loss = trainer.train(epoch, 0.0, 0.0, kbar)

        # Aggregate losses across distributed processes
        train_loss_tensor = torch.tensor([train_loss], dtype=torch.float).to(rank)
        dist.all_reduce(train_loss_tensor)
        ave_train_loss = train_loss_tensor.item() / len(train_data.dataset)

        val_loss_tensor = torch.tensor([val_loss], dtype=torch.float).to(rank)
        dist.all_reduce(val_loss_tensor)
        ave_val_loss = val_loss_tensor.item() / len(val_data.dataset)

        # Log metrics to wandb
        if rank == 0:
            wandb.log({
                "Validation F1 score": F1_score, "Validation precision": mAP,
                "Validation recall": mAR, "Training loss": ave_train_loss,
                "Validation loss": ave_val_loss,
            }, step=epoch)

        # Save model checkpoint
        name_output_file = (
            f"{config['name']}_epoch{epoch:02d}_loss_{ave_train_loss:.4f}_AP_{mAP:.4f}_"
            f"AR_{mAR:.4f}_trialnumber_{trial.number:02d}_batch"
            f"{config['dataloader']['train']['batch_size']:02d}_mimo{optuna_config['model']['mimo_layer']:02d}.pth"
        )
        filename = output_folder / exp_name / name_output_file
        checkpoint = {
            "net_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch_size": config['dataloader']['train']['batch_size'],
            "mimo_layer": optuna_config['model']['mimo_layer'],
            "lr": optuna_config['optimizer']['lr'],
            "history": history,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    if rank == 0:
        wandb.run.summary["final accuracy"] = F1_score
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

    return F1_score


def run_optimize(rank, world_size, return_dict, N_trials, config):
    """
    Entry point for running the Optuna optimization with distributed training.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        return_dict (multiprocessing.Manager.dict): A shared dictionary to store the study results.
        N_trials (int): Number of trials for Optuna optimization.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    print(f"Running distributed training on rank {rank} with {N_trials} trials.")
    ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    start_time = time.time()

    if rank == 0:
        study = optuna.create_study(
            direction="maximize",
            study_name="FFTRadNet_optimization",
            pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
        )
        study.optimize(
            partial(objective, config=config, rank=rank, world_size=world_size),
            n_trials=N_trials,
            timeout=300,
        )
        return_dict["study"] = study
    else:
        for _ in range(N_trials):
            try:
                objective(None, config, rank, world_size)
            except optuna.TrialPruned:
                pass

    print(f"Distributed training completed in {time.time() - start_time:.2f} seconds.")
    destroy_process_group()


if __name__ == "__main__":
    """
    Main entry point for running distributed training and Optuna optimization.

    This script performs the following:
    1. Parses command-line arguments for configuration, snapshot frequency, and the number of trials.
    2. Loads the configuration file.
    3. Sets random seeds for reproducibility.
    4. Sets up a distributed training environment across available GPUs.
    5. Runs the Optuna optimization process using distributed training.
    6. Collects and logs the study results, including statistics, best trial details, and saves a CSV summary.

    Command-line Arguments:
        -c, --config: Path to the configuration file (default: 'config.json').
        save_every: How often to save a snapshot.
        --trials: Number of Optuna trials to run (default: 50).
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simple distributed training job")
    parser.add_argument(
        "-c", "--config", default="config.json", type=str,
        help="Path to the config file (default: config.json)"
    )
    parser.add_argument(
        "save_every", type=int,
        help="How often to save a snapshot"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    args = parser.parse_args()

    # Load configuration from JSON file
    config = json.load(open(args.config))

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Prevent wandb initialization errors
    os.environ["WANDB_START_METHOD"] = "thread"

    # Determine the number of available GPUs for distributed training
    world_size = torch.cuda.device_count()

    # Shared dictionary for collecting results across processes
    manager = mp.Manager()
    return_dict = manager.dict()

    # Spawn processes for distributed training
    mp.spawn(
        run_optimize,  # Function to execute in each process
        args=(world_size, return_dict, args.trials, config),  # Arguments for `run_optimize`
        nprocs=world_size,  # Number of processes
        join=True  # Wait for all processes to finish
    )

    # Retrieve the study object from the shared dictionary
    study = return_dict["study"]

    # Analyze and log study results
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    best_trial = study.best_trial

    # Write summary statistics and best trial details to a file
    output_file = "optuna_paramter_tuning"
    with open(output_file, "a") as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")
        f.write("Best trial:\n")
        f.write(f"  Value: {best_trial.value}\n")
        f.write("  Params:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")

    # Save the study results to a CSV file
    study.trials_dataframe().to_csv("optuna_study.csv")
