import os
import torch
import torch.nn as nn
import numpy as np
#from some_module import YourDatasetClass, YourModelClass, FocalLoss  # Replace with your actual module and class names

import random
import json
import argparse
from model.FFTRadNet import FFTRadNet
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
from utils.evaluation import run_FullEvaluation
from loss import pixor_loss
from dataset.matlab_dataset import MATLAB
from utils.evaluation import run_evaluation

import matplotlib.pyplot as plt

def load_model_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net_state_dict'])
    return model


def plot_validation_losses(epoch_losses, output_file='validation_loss_plot.png'):
    epochs = sorted(epoch_losses.keys())
    losses = [epoch_losses[epoch] for epoch in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss per Epoch')
    plt.grid(True)
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')

def main(config, checkpoint_dir, output_file='validation_losses.txt', plot_file='validation_loss_plot.png'):
    
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)
    
    dataset = MATLAB(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

    # Initialize your model
    net = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = config['model']['MIMO_output'],
                        channels = config['model']['channels'], 
                        regression_layer = 2, 
                        detection_head = config['model']['DetectionHead'], 
                        segmentation_head = config['model']['SegmentationHead'])
    net = net.to('cuda')
    
    # Dictionary to store validation loss for each epoch
    epoch_losses = {}
    
    # Get list of checkpoint files
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    
    for checkpoint_file in checkpoint_files:
        # Extract epoch number from the filename
        epoch = int(checkpoint_file.split('_')[-1].split('.')[0])  # Assuming filename format like 'checkpoint_epoch_10.pth'
        
        # Load model from checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        print("checkpoint_path : ", checkpoint_path) 
        net = load_model_from_checkpoint(net, checkpoint_path)
        
        # Calculate validation loss
        #validation_loss = calculate_validation_loss(model, validation_loader, loss_fn)
        validation_loss = run_evaluation(net,val_loader,enc,check_perf=False,
                                detection_loss=pixor_loss,segmentation_loss=None,
                                losses_params=config['losses'])
        epoch_losses[epoch] = validation_loss
        print(f'Epoch {epoch}: Validation Loss = {validation_loss}')
    
    # Save the epoch losses to a file
    with open(output_file, 'w') as f:
        for epoch, loss in epoch_losses.items():
            f.write(f'Epoch {epoch}: {loss}\n')
    
    # Plot the validation losses
    plot_validation_losses(epoch_losses, plot_file)
    print("finish plotting")

if __name__ == "__main__":
    #checkpoint_dir = "/imec/other/dl4ms/chu06/RADIal/"  # Replace with your actual path
    #validation_data_path = "path/to/validation_data"  # Replace with your actual path
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Model checking')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpointdir', default=None, type=str,
                        help='Path to the .pth model checkpoint folder to load all checkpoint files')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpointdir)
    #main(checkpoint_dir, validation_data_path)
