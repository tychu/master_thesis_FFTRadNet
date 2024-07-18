import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
from model.FFTRadNet import FFTRadNet
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


def train(config, net, train_loader, optimizer, scheduler, history, kbar):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = net(inputs)

        classif_loss, reg_loss = pixor_loss(outputs['Detection'], label_map, config['losses'])
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

    return loss

    

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
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode)
    # Create the model
    # Suggest value for mimo_layer
    mimo_layer = trial.suggest_int('mimo_layer', 132, 192, step=12)

    net = FFTRadNet(
        blocks=config['model']['backbone_block'],
        mimo_layer=mimo_layer, #config['model']['MIMO_output'],
        channels=config['model']['channels'],
        regression_layer=2,
        detection_head=config['model']['DetectionHead'],
        segmentation_head=config['model']['SegmentationHead']
    )
    net.to('cuda')

    # Define hyperparameters to be tuned
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    step_size = trial.suggest_int('step_size', 5, 25, step=5) # paper set :10
    gamma =  float(config['lr_scheduler']['gamma']) #trial.suggest_uniform('gamma', 0.1, 0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Conditional parameter suggestion
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 20])
    if batch_size == 4 or batch_size == 8:
        num_epochs = 100
    elif batch_size == 16 or batch_size == 20:
        num_epochs = 200

    #num_epochs = int(config['num_epochs'])

    #freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'mIoU': []}

    train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])

    # init tracking experiment.
    # hyper-parameters, trial id are stored.
    # config = dict(trial.params)
    # config["trial.number"] = trial.number
    # wandb.init(
    #     project="optuna",
    #     entity="chu06",  # NOTE: this entity depends on your wandb account.
    #     config=config,
    #     group='FFTRadNet_optimization',
    #     reinit=True,
    # )

    for epoch in range(num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        loss = train(config, net, train_loader, optimizer, scheduler, history, kbar)

        eval = run_evaluation(trial, net, val_loader, enc, check_perf=(epoch >= 10),
                              detection_loss=pixor_loss, segmentation_loss=None,
                              losses_params=config['losses'])
        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])

        if eval['mAP'] + eval['mAR'] == 0:
            F1_score = 0
        else:
            F1_score = (eval['mAP']*eval['mAR'])/((eval['mAP'] + eval['mAR'])/2)


        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

        # Pruning
        trial.report(F1_score, epoch)
        # report F1_score to wandb
        wandb.log(data={"F1 score": F1_score, "precision":eval['mAP'], "recall":eval['mAR']}, step=epoch)

        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_trialnumber_{:02d}.pth'.format(epoch, loss, eval['mAP'], eval['mAR'], trial.number)
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



    return F1_score, eval['mAP'], eval['mAR']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFTRadNet Training with Optuna')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    config = json.load(open(args.config))

    # wandb might cause an error without this.
    #os.environ["WANDB_START_METHOD"] = "thread"
    wandb_kwargs = {"project": "optuna", "entity":"chu06-imec"}
    wandbc = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, config), n_trials=args.trials, callbacks=[wandbc])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Optionally save the study
    study.trials_dataframe().to_csv('optuna_study.csv')
