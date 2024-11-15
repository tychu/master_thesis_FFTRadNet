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
    net.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = net(inputs)

        classif_loss, reg_loss = pixor_loss(outputs, label_map, config['losses'])
        classif_loss *= config['losses']['weight'][0]
        reg_loss *= config['losses']['weight'][1]
        loss = classif_loss + reg_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        #kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()) ] )

    scheduler.step()
    history['train_loss'].append(running_loss / len(train_loader.dataset))
    history['lr'].append(scheduler.get_last_lr()[0])

    return running_loss / len(train_loader.dataset), outputs, label_map

    

def objective(trial, config):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)
    dataset = MATLAB(root_dir=config['dataset']['root_dir'],
                     folder_dir = config['dataset']['data_folder'], 
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode)
    # Create the model
    # Suggest value for mimo_layer
    mimo_layer = trial.suggest_int('mimo_layer', 64, 192, step=64)
    detection_head_layers = trial.suggest_categorical('detection_head_layers', [4])
    print('detection_head_layers: ', detection_head_layers)

    net = FFTRadNet(
        blocks=config['model']['backbone_block'],
        mimo_layer= mimo_layer,
        Ntx = config['model']['NbTxAntenna'],
        Nrx = config['model']['NbRxAntenna'],
        channels= config['model']['channels'],
        regression_layer=2,
        DH_num_layers = detection_head_layers, # number of detection_head layers, default: 4
        detection_head=config['model']['DetectionHead']
    )
    net.to('cuda')
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    print(net)

    # Define hyperparameters to be tuned
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # paper set :1e-4 lr = float(config['optimizer']['lr'])
    step_size = trial.suggest_int('step_size', 5, 20, step=5) # paper set :10 step_size = int(config['lr_scheduler']['step_size'])
    gamma =  float(config['lr_scheduler']['gamma']) #trial.suggest_uniform('gamma', 0.1, 0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Conditional parameter suggestion
    batch_size = 4 #trial.suggest_categorical('batch_size', [4]) # for testing script on not specific GPU server 
    if batch_size == 4 or batch_size == 8:
        num_epochs = 100
    elif batch_size == 16 or batch_size == 32:
        num_epochs = 200
    


    threshold = 0.2 #trial.suggest_categorical('threshold', [0.2]) # paper set 0.2
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'val_f1': [], 'train_f1': []}

    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])
    
    # init tracking experiment.
    # hyper-parameters, trial id are stored.
    config_optuna = dict(trial.params)
    config_optuna["FPN_channels"] = config['model']['channels']
    config_optuna["number_backbone_blocks"] = config['model']['backbone_block']
    config_optuna["detection_head_layer"] = detection_head_layers
    config_optuna["batch_size"] = batch_size
    config_optuna["trial.number"] = trial.number
    wandb.init(
        project=config['optuna_project'],
        entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
        config=config_optuna,
        group='FFTRadNet_optimization',
        reinit=True,
    )

    for epoch in range(num_epochs): #num_epochs
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        loss,predictions, ground_truth  = train(config, net, train_loader, optimizer, scheduler, history, kbar)

        
        # tra = run_evaluation(trial, net, train_loader, enc, threshold, check_perf=(epoch >= 1),
        #                       detection_loss=pixor_loss, segmentation_loss=None,
        #                       losses_params=config['losses'])
        
        eval = run_evaluation(trial, net, val_loader, enc, threshold, check_perf=(epoch >= 1),
                              detection_loss=pixor_loss, segmentation_loss=None,
                              losses_params=config['losses'])
        history['val_loss'].append(eval['loss']/ len(val_loader.dataset))
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])

        if eval['mAP'] + eval['mAR'] == 0:
            F1_score = 0
        else:
            F1_score = (eval['mAP']*eval['mAR'])/((eval['mAP'] + eval['mAR'])/2)
        
        # if tra['mAP'] + tra['mAR'] == 0:
        #     tra_F1_score = 0
        # else:
        #     tra_F1_score = (tra['mAP']*tra['mAR'])/((tra['mAP'] + tra['mAR'])/2)
        
        history['val_f1'].append(F1_score)
        #history['train_f1'].append(tra_F1_score)

        #kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

        # Pruning
        trial.report(F1_score, epoch)
        # report F1_score to wandb
        wandb.log(data={#"Train F1 score": tra_F1_score,
                        "validation F1 score": F1_score, 
                        "validation precision":eval['mAP'], 
                        "validation recall":eval['mAR'], 
                        "Training loss":loss, 
                        "Validation loss":eval['loss']/ len(val_loader.dataset),
                                                  }, 
                        step=epoch)

        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_trialnumber_{:02d}_batch{:02d}.pth'.format(epoch, loss, eval['mAP'], eval['mAR'], trial.number, batch_size)
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_size'] = batch_size
        #checkpoint['mimo_layer'] = mimo_layer
        checkpoint['lr'] = lr
        checkpoint['step_size'] = step_size
        checkpoint['history'] = history
        checkpoint['detectionhead_output'] = predictions


        
        # Save the state dictionary as a .pth file
        #torch.save(state_dict, filename)
        torch.save(checkpoint,filename)

        # Define the tar.gz file name
        tar_gz_filename = filename.with_suffix('.tar.gz')

        # Create a tar.gz file and add the .pth file to it
        with tarfile.open(tar_gz_filename, 'w:gz') as tarf:
            tarf.add(filename, arcname=name_output_file)

        # Optionally, you can remove the original .pth file if you only want the tar.gz file
        filename.unlink()

        print(f"Model saved and tarred with gzip compression as: {tar_gz_filename}")          
        print('')
    
    # report the final validation accuracy to wandb
    wandb.run.summary["final accuracy"] = eval['mAR']
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)



    return F1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFTRadNet Training with Optuna')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    config = json.load(open(args.config))

    # wandb might cause an error without this.
    os.environ["WANDB_START_METHOD"] = "thread"
    


    # Specific parameter combination for the first trial
    fixed_params = {
        "detection_head_layers" : 4,
        "lr": 1.46e-3,
        "step_size": 10,
        "mimo_layer": 128,
        
    }

    # Create a FixedTrial object
    fixed_trial = optuna.trial.FixedTrial(fixed_params)

    start_time = time.time()
    # Evaluate the objective function with the fixed parameters
    baseline_score = objective(fixed_trial, config)

    # Create a study
    # optuna without parallel
    study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', 
                                pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=5,
                                           n_warmup_steps=30, interval_steps=10))
    
    #study = optuna.create_study(direction='maximize')
    multi_gpu_time = time.time() - start_time
    #print(f"2 GPU training time: {multi_gpu_time:.2f} seconds\n")

    # Add the baseline trial to the study
    study.add_trial(optuna.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=baseline_score,
        params=fixed_params,
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "step_size": optuna.distributions.IntDistribution(5, 20, step=5),
            "mimo_layer": optuna.distributions.IntDistribution(64, 192, step=64),
            "detection_head_layers" : optuna.distributions.IntDistribution(1, 4, step=1)
        }
    ))
 
    study.optimize(lambda trial: objective(trial, config), n_trials=args.trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    trial = study.best_trial
    output_file = 'optuna_paramter_tuning'
    # Open the file in append mode and write the required information
    with open(output_file, 'a') as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")
        f.write("Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

    # Optionally save the study
    study.trials_dataframe().to_csv('optuna_study.csv')
