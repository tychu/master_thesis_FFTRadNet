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

from utils.plots import plot_histograms, create_video_from_images

# def load_model_from_checkpoint(model, checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['net_state_dict'])

#     return model

def read_epoch_losses(file_path, evalmode):
    epoch_losses = {}
    with open(file_path, 'r') as f:
        for line in f:
            if evalmode:
                # Split the line to extract epoch and loss
                parts = line.split(':')
                epoch = int(parts[0].strip().split()[1])
                loss = float(parts[1].strip())
                
                epoch_losses[epoch] = loss
            else:
                # Example line format: "Epoch 0: {'loss': 2562296.7998046875, 'mAP': 0, 'mAR': 0, 'mIoU': 0}"
                parts = line.strip().split(':', 1)
                epoch = int(parts[0].split()[1])
                
                # Find the loss value in the string
                loss_str = parts[1]
                loss = float(loss_str.split("'loss': ")[1].split(",")[0].strip())
                
                epoch_losses[epoch] = loss
    return epoch_losses

def load_all_predictions(checkpoint_dir):
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    predictions = []

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        prediction = checkpoint['prediction']  # Assuming prediction is stored in the checkpoint
        predictions.append(prediction)
    
    return predictions

def plot_validation_losses(epoch_train_losses, epoch_val_losses, val_division, evalmode, output_file):
    if evalmode:
        train_epochs = list(range(len(epoch_train_losses)))
        train_losses = epoch_train_losses
        
        #print("epoch_val_losses : ", epoch_val_losses)
        val_epochs = list(range(len(epoch_val_losses)))
        val_losses = [loss / val_division for loss in epoch_val_losses]
        #val_epochs = sorted(epoch_val_losses.keys())
        #val_losses = [epoch_val_losses[epoch] / val_division for epoch in val_epoch]
        #print(val_losses)
    else:
        train_epochs = list(range(len(epoch_train_losses)))
        train_losses = epoch_train_losses
        
        val_epochs = sorted(epoch_val_losses.keys())
        val_losses = [epoch_val_losses[epoch] / val_division for epoch in val_epochs]
        #print("epoch_val_losses : ", epoch_val_losses)
        #print("val_losses : ", val_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, color='blue', label='Training Loss')
    plt.plot(val_epochs, val_losses, color='green', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 800)
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    # Adding ticks every 10 epochs
    max_epoch = max(train_epochs + val_epochs)
    ticks = list(range(0, max_epoch + 1, 10))
    plt.xticks(ticks, rotation=45) 
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')

def main(config, checkpoint_dir, evalmode, histogram, lossplot, output_val_file='16rx_3targets_seqdata_eval_validation_losses_90epoch.txt', plot_file='16rx_3targets_seqdata_eval_validation_losses_90epoch.png'):
    #evalmode = True
    output_dir = "/imec/other/dl4ms/chu06/RADIal/FFTRadNet/plot/histogram/"

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

    val_division = len(val_loader.dataset)

    if evalmode: # already saved the loss in checkpoint_file.pth
        dict = torch.load(evalmode) 
        history = dict['history']
        epoch_train_losses = history['train_loss']
        epoch_val_losses = history['val_loss']

    else:       
        if not os.path.exists(output_val_file) or histogram:
            print("initialize model")
            # Initialize your model
            net = FFTRadNet(blocks = config['model']['backbone_block'],
                                mimo_layer  = config['model']['MIMO_output'],
                                channels = config['model']['channels'], 
                                regression_layer = 2, 
                                detection_head = config['model']['DetectionHead'], 
                                segmentation_head = config['model']['SegmentationHead'])
            net = net.to('cuda')
            
            # Dictionary to store validation loss for each epoch
            epoch_losses_val = {}
            epoch_losses_train = {}
            
            # Get list of checkpoint files
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
            
            for checkpoint_file in checkpoint_files:
                # Extract epoch number from the filename
                epoch = int(checkpoint_file.split('_')[2][5:])  # Extract the number after 'epoch'
                
                # Load model from checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                print("checkpoint_path : ", checkpoint_path) 
                #net = load_model_from_checkpoint(net, checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                net.load_state_dict(checkpoint['net_state_dict'])
                # 
                #prediction = checkpoint['prediction']  # Assuming prediction is stored in the checkpoint
                #predictions.append(prediction)

                if histogram:
                    if histogram == "train":
                        dataset = train_loader
                    elif histogram == 'val':
                        dataset = val_loader
                    for i, data in enumerate(dataset): # change to adjustable
                        if (i % 5 == 0):
                            inputs = data[0].to('cuda').float()


                            with torch.set_grad_enabled(False):
                                outputs = net(inputs)
                            
                            plot_histograms(outputs['Detection'], epoch, histogram, i,output_dir)

                
                if lossplot:
                    # Calculate validation loss without net.eval()
                    validation_loss = run_evaluation(net,val_loader,enc,check_perf=False,
                                            detection_loss=pixor_loss,segmentation_loss=None,
                                            losses_params=config['losses'])
                        
                    epoch_losses_val[epoch] = validation_loss 
                    print(f'Epoch {epoch}: Validation Loss = {validation_loss}')


            if lossplot:
                # Save the epoch losses to a file
                with open(output_val_file, 'w') as f:
                    for epoch, loss in epoch_losses_val.items():
                        f.write(f'Epoch {epoch}: {loss}\n')
                # with open(output_train_file, 'w') as f:
                #     for epoch, loss in epoch_losses_train.items():
                #         f.write(f'Epoch {epoch}: {loss}\n')
            if histogram:
                create_video_from_images(output_dir, fps=2)        
        else:
            print(f'{output_val_file} already exists. Skipping reading .pth files.')
        
        if lossplot:
        # Load model from checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'FFTRadNet_matlab_epoch499_loss_13.0259_AP_0.0000_AR_0.0000.pth')
            print("checkpoint_path : ", checkpoint_path) 
            dict = torch.load(checkpoint_path)
            history = dict['history']
            epoch_train_losses = history['train_loss']

            epoch_val_losses = read_epoch_losses(output_val_file, evalmode)  

    if lossplot:
        # Plot the validation losses

        plot_validation_losses(epoch_train_losses, epoch_val_losses, val_division, evalmode, plot_file)
        print("finish plotting")

if __name__ == "__main__":
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Model checking')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpointdir', default=None, type=str,
                        help='Path to the .pth model checkpoint folder to load all checkpoint files')
    parser.add_argument('--evalmode', default=None, type=str, help='Flag to use eval() and path to the checkpoint file')
    # the last epoch checkpoint file (have all the previous epoch record)
    parser.add_argument('--histogram', type=str, 
                        help='If provided, process data to create histograms. Provide string to indicate dataset.') # train_loader val_loader
    parser.add_argument('--lossplot', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpointdir, args.evalmode, args.histogram, args.lossplot)
    #main(checkpoint_dir, validation_data_path)
