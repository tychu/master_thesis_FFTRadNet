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

from model.FFTRadNet_ddp import FFTRadNet

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

        kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()) ] )

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)
    dataset = MATLAB(root_dir=config['dataset']['root_dir'],
                     folder_dir = config['dataset']['data_folder'], 
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode)
    # Create the model
    # Suggest value for mimo_layer
    mimo_layer = config['model']['MIMO_output'] #trial.suggest_int('mimo_layer', 64, 192, step=64)

    net = FFTRadNet(
        blocks=config['model']['backbone_block'],
        mimo_layer= mimo_layer,
        Ntx = config['model']['NbTxAntenna'],
        Nrx = config['model']['NbRxAntenna'],
        channels=config['model']['channels'],
        regression_layer=2,
        detection_head=config['model']['DetectionHead']
    )
    net.to('cuda')

    # Define hyperparameters to be tuned
    #lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # paper set :1e-4
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    #step_size = trial.suggest_int('step_size', 5, 15, step=5) # paper set :10
    gamma =  float(config['lr_scheduler']['gamma']) #trial.suggest_uniform('gamma', 0.1, 0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Conditional parameter suggestion
    #batch_size = trial.suggest_categorical('batch_size', [16, 32])
    batch_size = trial.suggest_categorical('batch_size', [4, 8]) # for testing script on not specific GPU server 
    if batch_size == 4 or batch_size == 8:
        num_epochs = 100
    elif batch_size == 16 or batch_size == 32:
        num_epochs = 200
    

    #num_epochs = int(config['num_epochs'])
    #threshold = trial.suggest_float("FFT_confidence_threshold", 0.1, 0.2, step=0.05) # paper set 0.2
    threshold =0.2

    #freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'val_f1': [], 'train_f1': []}

    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])
    
    # init tracking experiment.
    # hyper-parameters, trial id are stored.
    config_optuna = dict(trial.params)
    config_optuna["trial.number"] = trial.number
    wandb.init(
        project=config['optuna_project'],
        entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
        config=config_optuna,
        group='FFTRadNet_optimization',
        reinit=True,
    )

    for epoch in range(num_epochs):
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

        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

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

        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_trialnumber_{:02d}_batch{:02d}_mimo{:02d}.pth'.format(epoch, loss, eval['mAP'], eval['mAR'], trial.number, batch_size, mimo_layer)
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_size'] = batch_size
        checkpoint['mimo_layer'] = mimo_layer
        checkpoint['lr'] = lr
        checkpoint['step_size'] = step_size
        checkpoint['history'] = history
        checkpoint['detectionhead_output'] = predictions


        torch.save(checkpoint,filename)
            
        print('')
        # # Early stopping
        # if eval['mAP'] > best_mAP:
        #     best_mAP = eval['mAP']
        #     trial.set_user_attr('best_epoch', epoch)
        #     trial.set_user_attr('best_model', net.state_dict())
    
    # report the final validation accuracy to wandb
    wandb.run.summary["final accuracy"] = F1_score
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
    
    # optuna without parallel
    study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', pruner=optuna.pruners.MedianPruner())

    ## parallel optuna trial 
    # first command in terminal to initializing the study:
    # optuna create-study --study-name FFTRadNet_optimization --storage sqlite:///FFTRadNet_optimization_study.db
    # study = optuna.load_study(
    #     study_name="FFTRadNet_optimization", storage="sqlite:///FFTRadNet_optimization_study.db"
    # )
 
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
