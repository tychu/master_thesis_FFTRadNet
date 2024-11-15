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

# setting master port and address
def get_master_addr():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print("Master IP address: ", ip_address)
        return ip_address
    except socket.error as e:
        print(f"Failed to get IP address: {e}")
        raise

def find_free_port():
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
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    try:
        master_addr = get_master_addr()
        os.environ['MASTER_ADDR'] = master_addr


        #master_port = find_free_port()
        #os.environ['MASTER_PORT'] = '48113' # GPU v100
        os.environ['MASTER_PORT'] = '38853' # GPU a100


        start_time = time.time()

        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        end_time = time.time()
        init_duration = end_time - start_time
        print(f"Initialization took {init_duration:.2f} seconds.")
        
        print(f"Process {rank}/{world_size} initialized.")
        torch.cuda.set_device(rank)
    except Exception as e:
        print(f"Failed to set up DDP: {e}")
        raise


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data:DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        config:dict,
        optuna_config:dict,
        scheduler,
        encoder, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.config = config
        self.optuna_config = optuna_config
        self.scheduler = scheduler
        self.encoder = encoder
        
    def _compute_f1_score(self, TP, FP, FN):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    def _aggregate_metrics(self, local_TP, local_FP, local_FN, device):
        TP_tensor = torch.tensor([local_TP], dtype=torch.float).to(device)
        FP_tensor = torch.tensor([local_FP], dtype=torch.float).to(device)
        FN_tensor = torch.tensor([local_FN], dtype=torch.float).to(device)

        dist.all_reduce(TP_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(FP_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(FN_tensor, op=dist.ReduceOp.SUM)

        global_TP = TP_tensor.item()
        global_FP = FP_tensor.item()
        global_FN = FN_tensor.item()

        return global_TP, global_FP, global_FN
   
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            output = self.model(source)

        #loss = F.cross_entropy(output, targets)
        classif_loss,reg_loss = pixor_loss(output, targets, self.config['losses'])
        classif_loss *= self.config['losses']['weight'][0]
        reg_loss *= self.config['losses']['weight'][1]
        loss = classif_loss + reg_loss
        loss.backward()

        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch, total_train_loss, kbar):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        #for source, targets in self.train_data:
        for i, data in enumerate(self.train_data):
            source = data[0].to(self.gpu_id).float() #source.to(self.gpu_id)
            targets = data[1].to(self.gpu_id).float() #targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            # accumulate batch loss
            total_train_loss += loss.item()

            #kbar.update(i, values=[("loss", loss.item()) ] )
        
        self.scheduler.step() # lr_scheduler

        return total_train_loss

    def _run_train_evaluation(self, epoch):
        self.model.eval()

        train = run_evaluation(self.model, 
                                      self.train_data,
                                      self.encoder, 
                                      self.gpu_id, 
                                      self.optuna_config, 
                                      check_perf=(epoch >= 2),
                                      detection_loss=pixor_loss,
                                      losses_params=self.config['losses'])

        local_TP = train['TP']
        local_FP = train['FP']
        local_FN = train['FN']
        print("self.gpu_id: ", self.gpu_id)
        print("local_TP: ", local_TP)
        print("local_FP: ", local_FP)
        print("local_FN: ", local_FN)
        global_TP, global_FP, global_FN = self._aggregate_metrics(local_TP, local_FP, local_FN, self.gpu_id)
        F1_score, mAP, mAR = self._compute_f1_score(global_TP, global_FP, global_FN)


        return F1_score, mAP, mAR
    
    def _run_evaluation(self, epoch, total_val_loss):
        self.model.eval()

        eval = run_evaluation(self.model, 
                                      self.val_data,
                                      self.encoder, 
                                      self.gpu_id, 
                                      self.optuna_config, 
                                      check_perf=(epoch >= 2),
                                      detection_loss=pixor_loss,
                                      losses_params=self.config['losses'])

        local_TP = eval['TP']
        local_FP = eval['FP']
        local_FN = eval['FN']
        # print("self.gpu_id: ", self.gpu_id)
        # print("local_TP: ", local_TP)
        # print("local_FP: ", local_FP)
        # print("local_FN: ", local_FN)
        # global_TP, global_FP, global_FN = self._aggregate_metrics(local_TP, local_FP, local_FN, self.gpu_id)
        #F1_score, mAP, mAR = self._compute_f1_score(global_TP, global_FP, global_FN)
        F1_score, mAP, mAR = self._compute_f1_score(local_TP, local_FP, local_FN)


        print(f"Epoch {epoch} | Evaluation Loss: {eval['loss']/ len(self.val_data.dataset):.4f} | mAP: {eval['mAP']:.4f} | mAR: {eval['mAR']:.4f} | F1 Score: {F1_score:.4f}")
        total_val_loss += eval['loss']
        return F1_score, mAP, mAR, total_val_loss

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    #def train(self, max_epochs: int):
    def train(self, epoch: int, train_loss:int, val_loss:int, kbar):
        self.model.train()

        total_train_loss = self._run_epoch(epoch, train_loss, kbar)
        #f1_t, mAP_t, mAR_t =self._run_train_evaluation(epoch)
        f1, mAP, mAR, total_val_loss = self._run_evaluation(epoch, val_loss)
        if self.gpu_id == 0 and epoch % self.save_every == 0:
            self._save_checkpoint(epoch)

        return f1, mAP, mAR, total_val_loss, total_train_loss

# ddp
def load_train_objs(config, optuna_config):
    print("======== loading dataset and model =======")
    # load your dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                     statistics = config['dataset']['statistics'],
                     regression_layer = 2)
    dataset = MATLAB(root_dir = config['dataset']['root_dir'], 
                     folder_dir = config['dataset']['data_folder'], 
                     statistics= config['dataset']['statistics'],
                     encoder=enc.encode)
    

    # load your model
    model = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = optuna_config['model']['mimo_layer'], # optuna tuning
                        Ntx = config['model']['NbTxAntenna'],
                        Nrx = config['model']['NbRxAntenna'],
                        channels = config['model']['channels'], 
                        regression_layer = 2, 
                        detection_head = config['model']['DetectionHead'] 
                        )
    # Optimizer
    lr = optuna_config['optimizer']['lr'] # optuna tuning     #float(config['optimizer']['lr'])
    step_size = optuna_config['optimizer']['step_size'] #int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return dataset, model, optimizer, scheduler, enc

# ddp
def prepare_dataloader(config, dataset: Dataset, batch_size):
    print("======== preparing dataloader =======")
    train_loader, val_loader, _ = CreateDataLoaders(dataset,batch_size, config['dataloader'],config['seed'])

    # return DataLoader(dataset, batch_size=batch_size,pin_memory=True,shuffle=False, sampler=DistributedSampler(dataset))
    return train_loader, val_loader

def print_memory_summary():
    print("CUDA Memory Summary:")
    print(torch.cuda.memory_summary())
##########
# optuna #
##########
def objective(single_trial, config, rank,world_size):
    # saving checkpoint
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    trial = optuna.integration.TorchDistributedTrial(single_trial)
    
    # Set up the config as per trial parameters
    optuna_config = {
        "optimizer": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-4, log=True), # original: 1-e4
            'step_size': trial.suggest_categorical('step_size', [10]),
            #"step_size": trial.suggest_int('step_size', 5, 20, step=5) # original: 10
        },
        "model": {
            #"mimo_layer": trial.suggest_int('mimo_layer', 64, 192, step=64) # original: 192
            "mimo_layer": trial.suggest_categorical('mimo_layer', [192]) # original: 192
        },
        #"batch_size": trial.suggest_categorical('batch_size', [32]), # original: 4
        #"threshold":  trial.suggest_float("FFT_confidence_threshold", 0.1, 0.2, step=0.05) # original: 0.2
    }

    # load model and dataset
    dataset, model, optimizer, scheduler, encoder = load_train_objs(config, optuna_config)
    
    #train_data, val_data = prepare_dataloader(config, dataset, optuna_config['batch_size'])
    train_data, val_data = prepare_dataloader(config, dataset, config['dataloader']['train']['batch_size'])

    # init trainer
    trainer = Trainer(model, train_data, val_data, optimizer, rank, config['save_every'], config, optuna_config, scheduler, encoder)

    # set epoch
    num_epochs = 5 #100
    # if optuna_config['batch_size'] == 4 or optuna_config['batch_size'] == 8:
    #     num_epochs = 100
    # elif optuna_config['batch_size'] == 16 or optuna_config['batch_size'] == 32:
    #     num_epochs = 200 

    # wandb
    # init tracking experiment
    # hyper-parameters, trial id are stored.
    if rank == 0:
        config_optuna = dict(trial.params)
        config_optuna["trial.number"] = trial.number
        wandb.init(
            project=config['optuna_project'],
            entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
            config=config_optuna,
            group='FFTRadNet_optimization',
            reinit=True,
        )
    # wandb 
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'F1_score': [], 
               'train_mAP': [], 'train_mAR': [], 'train_F1_score': []}


    print(range(num_epochs))
    for epoch in range(num_epochs):
        # Initialize epoch_loss to accumulate the loss over all batches in this epoch.
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        

        
        print('--- epoch: ', epoch)
        kbar = pkbar.Kbar(target=len(train_data), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        # Training the model and compute evaluate score
        F1_score, mAP, mAR, val_loss, train_loss = trainer.train(epoch, epoch_train_loss, epoch_val_loss, kbar)



        # Use torch.distributed.all_reduce to sum up the losses across all processes.
        train_loss_tensor = torch.tensor([train_loss], dtype=torch.float).to(rank)
        #dist.all_reduce(train_loss_tensor)
        ave_train_loss = train_loss_tensor.item() / (len(train_data.dataset)/world_size)  # Optionally average the loss
        val_loss_tensor = torch.tensor([val_loss], dtype=torch.float).to(rank)
        #dist.all_reduce(val_loss_tensor)
        ave_val_loss = val_loss_tensor.item() / (len(val_data.dataset)/world_size)  # Optionally average the loss
        
        history['train_loss'].append(train_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['val_loss'].append(val_loss)
        history['mAP'].append(mAP)
        history['mAR'].append(mAR)
        history['F1_score'].append(F1_score)


        print("trial.report")
        trial.report(F1_score, epoch)
        if rank == 0:
            # report F1_score to wandb
            wandb.log(data={"Validation F1 score": F1_score, 
                            "Validation precision":mAP, 
                            "Validation recall":mAR, 
                            "Training loss":ave_train_loss, 
                            "Validation loss":ave_val_loss,
                                                    }, 
                            step=epoch)

        if trial.should_prune():
            if rank == 0:
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")
        
        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_trialnumber_{:02d}_batch{:02d}_mimo{:02d}.pth'.format(epoch
                                                                                                                                              , ave_train_loss
                                                                                                                                              , mAP
                                                                                                                                              , mAR
                                                                                                                                              , trial.number
                                                                                                                                              , config['dataloader']['train']['batch_size'] #optuna_config['batch_size']
                                                                                                                                              , optuna_config['model']['mimo_layer'])
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_size'] = config['dataloader']['train']['batch_size'] #optuna_config['batch_size']
        checkpoint['mimo_layer'] = optuna_config['model']['mimo_layer']
        checkpoint['lr'] = optuna_config['optimizer']['lr']
        checkpoint['history'] = history

        torch.save(checkpoint,filename)
        print("checkpoint filename: ", filename)
   
    if rank == 0:
        # report the final validation accuracy to wandb
        wandb.run.summary["final accuracy"] = F1_score
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)
    #destroy_process_group()

    return F1_score  # Replace with the metric you are optimizing


def run_optimize(rank, world_size, return_dict, N_trials, config):
    
    print(f"Running basic DDP example on rank {rank}")
    print(f"number of trial {N_trials}")

    ddp_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    start_time = time.time()

    if rank == 0:
        study = optuna.create_study(direction="maximize", study_name='FFTRadNet_optimization', 
                                pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=5,
                                           n_warmup_steps=30, interval_steps=10))
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

    multi_gpu_time = time.time() - start_time
    print(f"2 GPU training time: {multi_gpu_time:.2f} seconds\n")
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='Path to the config file (default: config.json)')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    config = json.load(open(args.config))
    
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    # wandb might cause an error without this.
    os.environ["WANDB_START_METHOD"] = "thread"

    world_size = torch.cuda.device_count()
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        run_optimize,
        args=(world_size, return_dict, args.trials, config),
        nprocs=world_size,
        join=True,
    )
    study = return_dict["study"]

        
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    btrial = study.best_trial
    output_file = 'optuna_paramter_tuning'
    # Open the file in append mode and write the required information
    with open(output_file, 'a') as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")
        f.write("Best trial:\n")
        f.write(f"  Value: {btrial.value}\n")
        f.write("  Params: \n")
        for key, value in btrial.params.items():
            f.write(f"    {key}: {value}\n")

    # Optionally save the study
    study.trials_dataframe().to_csv('optuna_study.csv')
