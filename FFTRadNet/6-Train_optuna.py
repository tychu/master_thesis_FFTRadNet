import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from dataset.dataset import RADIal
from dataset.matlab_dataset import MATLAB
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
from utils.evaluation import run_evaluation
import torch.nn as nn
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from model.FFTRadNet_redlay import FFTRadNet # can reduce layer not channels
#from model.FFTRadNet_ddp import FFTRadNet # reduce layer and specific channels
import time
import tarfile

def train(config, net, train_loader, optimizer, scheduler, history, kbar):
    """
    Perform one epoch of training for the model.

    Args:
        config (dict): Configuration dictionary.
        net (torch.nn.Module): Neural network to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
        scheduler (Scheduler): Learning rate scheduler.
        history (dict): Dictionary to store training history.
        kbar (Kbar): Progress bar for the training process.

    Returns:
        Tuple: Losses and model outputs for the epoch.
    """
    # Set the network to training mode
    net.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        # Move input data and labels to the GPU
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass with gradients enabled
        with torch.set_grad_enabled(True):
            outputs = net(inputs)

        # Compute classification and regression losses
        classif_loss, reg_loss = pixor_loss(outputs, label_map, config['losses'])
        classif_loss *= config['losses']['weight'][0]  # Weighted classification loss
        reg_loss *= config['losses']['weight'][1]      # Weighted regression loss
        loss = classif_loss + reg_loss

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    # Update the learning rate scheduler
    scheduler.step()

    # Log training loss and learning rate
    history['train_loss'].append(running_loss / len(train_loader.dataset))
    history['lr'].append(scheduler.get_last_lr()[0])

    return running_loss / len(train_loader.dataset), outputs, label_map

    

def objective(trial, config):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object.
        config (dict): Configuration dictionary.

    Returns:
        float: Final training loss or other evaluation metric.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Generate experiment name and create output directories
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

    # Save configuration for this experiment
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # Select device (GPU or CPU)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Initialize the encoder and dataset
    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)
    dataset = MATLAB(root_dir=config['dataset']['root_dir'],
                     folder_dir=config['dataset']['data_folder'], 
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode)

    # Define and initialize the model with trial-specific parameters
    mimo_layer = trial.suggest_int('mimo_layer', 64, 192, step=64)
    detection_head_layers = trial.suggest_categorical('detection_head_layers', [4])
    net = FFTRadNet(
        blocks=config['model']['backbone_block'],
        mimo_layer=mimo_layer,
        Ntx=config['model']['NbTxAntenna'],
        Nrx=config['model']['NbRxAntenna'],
        channels=config['model']['channels'],
        regression_layer=2,
        DH_num_layers=detection_head_layers,
        detection_head=config['model']['DetectionHead']
    )
    net.to('cuda')

    # Display model information
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ", t_params)
    print(net)

    # Define optimizer, learning rate scheduler, and batch size
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    step_size = trial.suggest_int('step_size', 5, 20, step=5)
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    batch_size = 4

    # Define training and evaluation hyperparameters
    num_epochs = 100 if batch_size in [4, 8] else 200
    threshold = 0.2
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'val_f1': [], 'train_f1': []}

    # Create data loaders for training and validation
    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])

    # Initialize experiment tracking (e.g., with WandB)
    config_optuna = dict(trial.params)
    wandb.init(
        project=config['optuna_project'],
        entity="chu06-imec",
        config=config_optuna,
        group='FFTRadNet_optimization',
        reinit=True,
    )

    # Training loop
    for epoch in range(num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        # Train and evaluate the model
        loss, predictions, ground_truth = train(config, net, train_loader, optimizer, scheduler, history, kbar)
        eval = run_evaluation(trial, net, val_loader, enc, threshold, check_perf=(epoch >= 1),
                              detection_loss=pixor_loss, segmentation_loss=None,
                              losses_params=config['losses'])

        # Log validation metrics
        history['val_loss'].append(eval['loss'] / len(val_loader.dataset))
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])
        F1_score = (eval['mAP'] * eval['mAR']) / ((eval['mAP'] + eval['mAR']) / 2) if eval['mAP'] + eval['mAR'] > 0 else 0
        history['val_f1'].append(F1_score)

        wandb.log({
            "validation F1 score": F1_score,
            "validation precision": eval['mAP'],
            "validation recall": eval['mAR'],
            "Training loss": loss,
            "Validation loss": eval['loss'] / len(val_loader.dataset),
        }, step=epoch)

        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        # Save the model and compress it into tar.gz
        name_output_file = f"{config['name']}_epoch{epoch:02d}_loss_{loss:.4f}_AP_{eval['mAP']:.4f}_AR_{eval['mAR']:.4f}_trialnumber_{trial.number:02d}_batch{batch_size:02d}.pth"
        filename = output_folder / exp_name / name_output_file
        checkpoint = {
            'net_state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'lr': lr,
            'step_size': step_size,
            'history': history,
            'detectionhead_output': predictions
        }
        torch.save(checkpoint, filename)

        tar_gz_filename = filename.with_suffix('.tar.gz')
        with tarfile.open(tar_gz_filename, 'w:gz') as tarf:
            tarf.add(filename, arcname=name_output_file)
        filename.unlink()

        print(f"Model saved and tarred with gzip compression as: {tar_gz_filename}")

    wandb.run.summary["final accuracy"] = eval['mAR']
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return F1_score

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='FFTRadNet Training with Optuna')
    # Argument for the configuration file path
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    # Argument for the number of Optuna trials
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Load the configuration file
    config = json.load(open(args.config))

    # Set up the environment variable to avoid potential issues with wandb
    os.environ["WANDB_START_METHOD"] = "thread"

    # Define specific parameter combinations for the first trial (fixed parameters)
    fixed_params = {
        "detection_head_layers": 4,  # Fixed number of detection head layers
        "lr": 1.46e-3,              # Fixed learning rate
        "step_size": 10,            # Fixed step size for learning rate scheduler
        "mimo_layer": 128,          # Fixed MIMO layer size
    }

    # Create a FixedTrial object for the baseline trial
    fixed_trial = optuna.trial.FixedTrial(fixed_params)

    # Start timer to measure execution time
    start_time = time.time()
    # Evaluate the objective function with the fixed parameters (baseline trial)
    baseline_score = objective(fixed_trial, config)

    # Create an Optuna study to optimize the objective function
    # Use PercentilePruner to prune trials based on performance
    study = optuna.create_study(
        direction='maximize',  # Objective is to maximize the evaluation metric
        study_name='FFTRadNet_optimization',  # Name of the study
        pruner=optuna.pruners.PercentilePruner(
            50.0,  # Prune trials below the 50th percentile
            n_startup_trials=5,  # Minimum number of trials before pruning
            n_warmup_steps=30,   # Number of epochs before pruning
            interval_steps=10    # Interval for checking pruning criteria
        )
    )

    # Measure and print the elapsed time for the baseline trial
    multi_gpu_time = time.time() - start_time

    # Add the baseline trial to the study
    study.add_trial(optuna.create_trial(
        state=optuna.trial.TrialState.COMPLETE,  # Mark trial as complete
        value=baseline_score,  # Baseline trial score
        params=fixed_params,  # Baseline trial parameters
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "step_size": optuna.distributions.IntDistribution(5, 20, step=5),
            "mimo_layer": optuna.distributions.IntDistribution(64, 192, step=64),
            "detection_head_layers": optuna.distributions.IntDistribution(1, 4, step=1),
        }
    ))

    # Optimize the study with a given number of trials
    study.optimize(lambda trial: objective(trial, config), n_trials=args.trials)

    # Retrieve pruned and complete trials for logging
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Get the best trial from the study
    trial = study.best_trial

    # Output file for logging study statistics and best trial details
    output_file = 'optuna_paramter_tuning'
    with open(output_file, 'a') as f:
        # Write study statistics
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")
        # Write best trial details
        f.write("Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

    # Optionally save the study results to a CSV file for further analysis
    study.trials_dataframe().to_csv('optuna_study.csv')
