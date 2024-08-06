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
from utils.evaluation import run_evaluation
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

from model.FFTRadNet_ddp import FFTRadNet
from dataset.dataloader_ddp import CreateDataLoaders
from dataset.matlab_dataset_ddp import MATLAB
from utils.evaluation import run_evaluation_ddp
from multiprocessing import Queue





# def cleanup():
#     dist.destroy_process_group()

# def train(config, net, train_loader, optimizer, scheduler, rank, history, kbar):
#     net.train()
#     running_loss = 0.0

#     for i, data in enumerate(train_loader):
#         #inputs = data[0].to('cuda').float()
#         #label_map = data[1].to('cuda').float()
#         inputs = data[0].to(rank).float()
#         label_map = data[1].to(rank).float()

#         optimizer.zero_grad()

#         with torch.set_grad_enabled(True):
#             outputs = net(inputs)

#         classif_loss, reg_loss = pixor_loss(outputs['Detection'], label_map, config['losses'])
#         classif_loss *= config['losses']['weight'][0]
#         reg_loss *= config['losses']['weight'][1]
#         loss = classif_loss + reg_loss
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)

#         kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()) ] )

#     scheduler.step()
#     history['train_loss'].append(running_loss / len(train_loader.dataset))
#     history['lr'].append(scheduler.get_last_lr()[0])

#     return loss, outputs['Detection'], label_map

    

# def objective(trial, config, rank, world_size):
#     torch.manual_seed(config['seed'])
#     np.random.seed(config['seed'])
#     random.seed(config['seed'])
#     torch.cuda.manual_seed(config['seed'])

#     # checkpoint filename
#     curr_date = datetime.now()
#     exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
#     output_folder = Path(config['output']['dir'])
#     output_folder.mkdir(parents=True, exist_ok=True)
#     (output_folder / exp_name).mkdir(parents=True, exist_ok=True)

#     with open(output_folder / exp_name / 'config.json', 'w') as outfile:
#         json.dump(config, outfile)


#     # setting GPU
#     #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     enc = ra_encoder(geometry=config['dataset']['geometry'],
#                      statistics=config['dataset']['statistics'],
#                      regression_layer=2)
#     dataset = MATLAB(root_dir=config['dataset']['root_dir'],
#                      statistics=config['dataset']['statistics'],
#                      encoder=enc.encode)
#     # Create the model
#     # Suggest value for mimo_layer
#     mimo_layer = trial.suggest_int('mimo_layer', 64, 192, step=64)

#     net = FFTRadNet(
#         blocks=config['model']['backbone_block'],
#         mimo_layer=mimo_layer, #config['model']['MIMO_output'],
#         channels=config['model']['channels'],
#         regression_layer=2,
#         detection_head=config['model']['DetectionHead'],
#         segmentation_head=config['model']['SegmentationHead']
#     )
#     #net.to('cuda')

#     # setup DDP
#     setup(rank, world_size)
#     model = net.to(rank)
#     ddp_model = DDP(model, device_ids=[rank])

#     # Define hyperparameters to be tuned
#     lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # paper set :1e-4
#     step_size = trial.suggest_int('step_size', 5, 15, step=5) # paper set :10
#     gamma =  float(config['lr_scheduler']['gamma']) #trial.suggest_uniform('gamma', 0.1, 0.9)
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#     # Conditional parameter suggestion
#     batch_size = trial.suggest_categorical('batch_size', [4, 16, 32])
#     #batch_size = trial.suggest_categorical('batch_size', [4]) # for testing script on not specific GPU server 
#     if batch_size == 4 or batch_size == 8:
#         num_epochs = 100
#     elif batch_size == 16 or batch_size == 32:
#         num_epochs = 200

#     #num_epochs = int(config['num_epochs'])

#     #freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')
#     history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'mIoU': []}

#     train_loader, val_loader, _ = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])

#     # init tracking experiment.
#     # hyper-parameters, trial id are stored.
#     config_optuna = dict(trial.params)
#     config_optuna["trial.number"] = trial.number
#     wandb.init(
#         project="optuna-largebatch",
#         entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
#         config=config_optuna,
#         group='FFTRadNet_optimization',
#         reinit=True,
#     )

#     for epoch in range(num_epochs):
#         kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

#         loss,predictions, ground_truth  = train(config, ddp_model, train_loader, optimizer, scheduler, rank, history, kbar)

#         eval = run_evaluation(trial, net, val_loader, enc, check_perf=(epoch >= 10),
#                               detection_loss=pixor_loss, segmentation_loss=None,
#                               losses_params=config['losses'])
#         history['val_loss'].append(eval['loss']/ len(val_loader.dataset))
#         history['mAP'].append(eval['mAP'])
#         history['mAR'].append(eval['mAR'])

#         if eval['mAP'] + eval['mAR'] == 0:
#             F1_score = 0
#         else:
#             F1_score = (eval['mAP']*eval['mAR'])/((eval['mAP'] + eval['mAR'])/2)


#         kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

#         # Pruning
#         trial.report(F1_score, epoch)
#         # report F1_score to wandb
#         wandb.log(data={"F1 score": F1_score, 
#                         "precision":eval['mAP'], 
#                         "recall":eval['mAR'], 
#                         "Training loss":loss, 
#                         "Validation loss":eval['loss'],
#                         # "pr": wandb.plot.pr_curve(ground_truth[:, 0,:, :].detach().cpu().numpy().copy().flatten(), 
#                         #                           predictions[:, 0,:, :].detach().cpu().numpy().copy().flatten())
#                                                   }, 
#                         step=epoch)

#         if trial.should_prune():
#             wandb.run.summary["state"] = "pruned"
#             wandb.finish(quiet=True)
#             raise optuna.exceptions.TrialPruned()

#         name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_trialnumber_{:02d}_batch{:02d}_mimo{:02d}.pth'.format(epoch, loss, eval['mAP'], eval['mAR'], trial.number, batch_size, mimo_layer)
#         filename = output_folder / exp_name / name_output_file

#         checkpoint={}
#         checkpoint['net_state_dict'] = net.state_dict()
#         checkpoint['optimizer'] = optimizer.state_dict()
#         checkpoint['scheduler'] = scheduler.state_dict()
#         checkpoint['epoch'] = epoch
#         checkpoint['batch_size'] = batch_size
#         checkpoint['mimo_layer'] = mimo_layer
#         checkpoint['lr'] = lr
#         checkpoint['step_size'] = step_size
#         checkpoint['history'] = history


#         torch.save(checkpoint,filename)
            
#         print('')
#         # # Early stopping
#         # if eval['mAP'] > best_mAP:
#         #     best_mAP = eval['mAP']
#         #     trial.set_user_attr('best_epoch', epoch)
#         #     trial.set_user_attr('best_model', net.state_dict())
    
#     # DDP
#     cleanup()
    
#     # report the final validation accuracy to wandb
#     wandb.run.summary["final accuracy"] = F1_score
#     wandb.run.summary["state"] = "completed"
#     wandb.finish(quiet=True)



#     return F1_score, eval['mAP'], eval['mAR']

# if __name__ == '__main__':
#     rank = 0  # Assuming this is the rank of the current process
#     world_size = 4  # Total number of processes
#     setup(rank, world_size)

    # parser = argparse.ArgumentParser(description='FFTRadNet Training with Optuna')
    # parser.add_argument('-c', '--config', default='config.json',type=str,
    #                     help='Path to the config file (default: config.json)')
    # parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    # args = parser.parse_args()
    # config = json.load(open(args.config))

    # # wandb might cause an error without this.
    # os.environ["WANDB_START_METHOD"] = "thread"
    
    # # optuna without parallel
    # study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', pruner=optuna.pruners.MedianPruner())

    # ## parallel optuna trial 
    # # first command in terminal to initializing the study:
    # # optuna create-study --study-name FFTRadNet_optimization --storage sqlite:///FFTRadNet_optimization_study.db
    # # study = optuna.load_study(
    # #     study_name="FFTRadNet_optimization", storage="sqlite:///FFTRadNet_optimization_study.db"
    # # )
 
    # study.optimize(lambda trial: objective(trial, config), n_trials=args.trials)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # trial = study.best_trial
    # output_file = 'optuna_paramter_tuning'
    # # Open the file in append mode and write the required information
    # with open(output_file, 'a') as f:
    #     f.write("Study statistics:\n")
    #     f.write(f"  Number of finished trials: {len(study.trials)}\n")
    #     f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
    #     f.write(f"  Number of complete trials: {len(complete_trials)}\n")
    #     f.write("Best trial:\n")
    #     f.write(f"  Value: {trial.value}\n")
    #     f.write("  Params: \n")
    #     for key, value in trial.params.items():
    #         f.write(f"    {key}: {value}\n")

    # # Optionally save the study
    # study.trials_dataframe().to_csv('optuna_study.csv')

# for queue and ddp to use the same sulitprocessing
mp.set_start_method('spawn', force=True)

# setting master port and address
def get_master_addr():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    #print("ip_address: ", ip_address)
    return ip_address

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        #print('free port: ', s.getsockname()[1])
        return s.getsockname()[1]

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = get_master_addr()
    os.environ['MASTER_PORT'] = str(find_free_port())

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        config: dict,
        optuna_config: dict,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        encoder, 
        trial,
        
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.config = config
        self.encoder = encoder
        self.threshold = optuna_config['threshold']
        self.trial = trial


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
        batch_loss += loss.item() * source.size(0)

        return batch_loss

    def _run_epoch(self, epoch, scheduler, epoch_loss):
        #b_sz = len(next(iter(self.train_data))[0])
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        print(f"Dataset length: {len(self.train_data.dataset)}")
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        #for source, targets in self.train_data:
        for i, data in enumerate(self.train_data):
            print(f"Batch {i}: ")

            source = data[0].to(self.gpu_id).float()
            targets = data[1].to(self.gpu_id).float()
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
            scheduler.step() # lr_scheduler

        return epoch_loss
    
    def _run_evaluation(self, epoch, encoder, threshold):
        self.model.eval()

        # with torch.no_grad():
        #     for i, data in enumerate(self.val_data):
        #         source = data[0].to(self.gpu_id).float()
        #         targets = data[1].to(self.gpu_id).float()
        #         output = self.model(source)

        eval = run_evaluation_ddp(self.trial, self.model, self.val_data, encoder, self.gpu_id, threshold, check_perf=(epoch >= 10),
                            detection_loss=pixor_loss, segmentation_loss=None,
                            losses_params=self.config['losses'])

        F1_score = 0 if (eval['mAP'] + eval['mAR']) == 0 else (eval['mAP']*eval['mAR']) / ((eval['mAP'] + eval['mAR']) / 2)

        self.history['val_loss'].append(eval['loss']/ len(self.val_data.dataset))
        self.history['mAP'].append(eval['mAP'])
        self.history['mAR'].append(eval['mAR'])

        print(f"Epoch {epoch} | Evaluation Loss: {eval['loss']/ len(self.val_data.dataset):.4f} | mAP: {eval['mAP']:.4f} | mAR: {eval['mAR']:.4f} | F1 Score: {F1_score:.4f}")

        return F1_score, eval['mAP'], eval['mAR'], eval['loss']/ len(self.val_data.dataset)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int, scheduler, trial):
        self.model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            # Run training
            running_loss = self._run_epoch(epoch, scheduler, epoch_loss)

            # Run evaluation
            F1_score, mAP, mAR, val_loss = self._run_evaluation(epoch, self.encoder, self.threshold)
            trial.report(F1_score, epoch)
            # report F1_score to wandb
            wandb.log(data={"F1 score": F1_score, 
                        "precision":mAP, 
                        "recall":mAR, 
                        "Training loss":running_loss, 
                        "Validation loss":val_loss}, 
                        step=epoch)
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()
        
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        
        # report the final validation accuracy to wandb
        wandb.run.summary["final accuracy"] = F1_score
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

        return F1_score

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
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return dataset, model, optimizer, scheduler, enc


def prepare_dataloader(config, dataset: Dataset, batch_size):
    print("======== preparing dataloader =======")
    train_loader, val_loader, _ = CreateDataLoaders(dataset,batch_size, config['dataloader'],config['seed'])

    # return DataLoader(dataset, batch_size=batch_size,pin_memory=True,shuffle=False, sampler=DistributedSampler(dataset))
    return train_loader, val_loader

# def test_getitem(dataset):
#     for idx in range(len(dataset)):
#         try:
#             radar_FFT, segmap, out_label, box_labels, image = dataset[idx]
#             print(f"Index {idx} processed successfully.")
#         except Exception as e:
#             print(f"Error at index {idx}: {e}")

# def main(rank: int, world_size: int, config, single_trial, queue):

#     print(f"Running main on rank {rank}")
#     #print(f"Configuration: {config}")
        
#     # Setup random seed
#     torch.manual_seed(config['seed'])
#     np.random.seed(config['seed'])
#     random.seed(config['seed'])
#     torch.cuda.manual_seed(config['seed'])

#     print('Main script and DDP setup')
#     ddp_setup(rank, world_size)

#     trial = optuna.integration.TorchDistributedTrial(single_trial)
#     # Set up the config as per trial parameters
#     optuna_config = {
#         "optimizer": {
#             "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True) # original: 1-e4
#         },
#         "model": {
#             "mimo_layer": trial.suggest_int('mimo_layer', 64, 192, step=64) # original: 192
#         },
#         "batch_size": trial.suggest_categorical('batch_size', [4, 8, 16, 32]), # original: 4
#         "threshold":  trial.suggest_float("FFT_confidence_threshold", 0.1, 0.5, step=0.1) # original: 0.2
#     }

#     dataset, model, optimizer, scheduler, encoder = load_train_objs(config, optuna_config)
#     train_data, val_data = prepare_dataloader(config, dataset, optuna_config['batch_size'])
    
#     # init
#     trainer = Trainer(config, optuna_config, model, train_data, val_data, optimizer, rank, config['save_every'], encoder)
#     # init tracking experiment.
#     # hyper-parameters, trial id are stored.
#     config_optuna = dict(trial.params)
#     config_optuna["trial.number"] = trial.number
#     wandb.init(
#         project=config['optuna_project'],
#         entity="chu06-imec",  # NOTE: this entity depends on your wandb account.
#         config=config_optuna,
#         group='FFTRadNet_optimization',
#         reinit=True,
#     )

#     if optuna_config['batch_size'] == 4 or optuna_config['batch_size'] == 8:
#         num_epochs = 100
#     elif optuna_config['batch_size'] == 16 or optuna_config['batch_size'] == 32:
#         num_epochs = 200
#     total_epochs = num_epochs
    
#     # train and compute evaluate score
#     F1_score = trainer.train(total_epochs, scheduler, trial)

#     # For Optuna
#     if rank == 0:
#         queue.put(F1_score)  # Put the result in the queue
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()

#     destroy_process_group()

##########
# optuna #
##########
def objective(single_trial, config):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Set up the distributed environment for the trial
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    ddp_setup(rank, world_size)
    print(f"Running main on rank {rank}")

    trial = optuna.integration.TorchDistributedTrial(single_trial)
    # Set up the config as per trial parameters
    optuna_config = {
        "optimizer": {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True) # original: 1-e4
        },
        "model": {
            "mimo_layer": trial.suggest_int('mimo_layer', 64, 192, step=64) # original: 192
        },
        "batch_size": trial.suggest_categorical('batch_size', [4, 8, 16, 32]), # original: 4
        "threshold":  trial.suggest_float("FFT_confidence_threshold", 0.1, 0.5, step=0.1) # original: 0.2
    }

    dataset, model, optimizer, scheduler, encoder = load_train_objs(config, optuna_config)
    train_data, val_data = prepare_dataloader(config, dataset, optuna_config['batch_size'])
    
    # init
    trainer = Trainer(config, optuna_config, model, train_data, val_data, optimizer, rank, config['save_every'], encoder)
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

    if optuna_config['batch_size'] == 4 or optuna_config['batch_size'] == 8:
        num_epochs = 100
    elif optuna_config['batch_size'] == 16 or optuna_config['batch_size'] == 32:
        num_epochs = 200
    total_epochs = num_epochs
    
    # train and compute evaluate score
    F1_score = trainer.train(total_epochs, scheduler, trial)

    # For Optuna
    if rank == 0:
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    destroy_process_group()

    return F1_score  # Replace with the metric you are optimizing

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='simple distributed training job')
    
#     parser.add_argument('-c', '--config', default='config.json',type=str,
#                         help='Path to the config file (default: config.json)')
    
#     parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
#     parser.add_argument('save_every', type=int, help='How often to save a snapshot')
#     parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 4)')
#     args = parser.parse_args()
    
#     try:
#         config = json.load(open(args.config))
#     except Exception as e:
#         print(f"Failed to load config file: {e}")
#         exit(1)
    
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, config), nprocs=world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='Path to the config file (default: config.json)')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    
    try:
        config = json.load(open(args.config))
    except Exception as e:
        print(f"Failed to load config file: {e}")
        exit(1)

    # wandb might cause an error without this.
    os.environ["WANDB_START_METHOD"] = "thread"

    world_size = torch.cuda.device_count()
    rank = os.getenv("OMPI_COMM_WORLD_RANK", None)
    if rank is None:
        rank = os.getenv("PMI_RANK", None)
    if rank is None:
        rank = 0  # Default rank to 0 if not set
    rank = int(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


    # Set up the distributed environment
    ddp_setup(int(rank), world_size)

    if rank == 0:
        study = optuna.create_study(direction='maximize', study_name='FFTRadNet_optimization', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, config), n_trials=args.trials)
        
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
    else:
        for _ in range(20):
            try:
                objective(None)
            except optuna.TrialPruned:
                pass