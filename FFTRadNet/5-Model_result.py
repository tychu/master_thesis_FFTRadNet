import os
import torch
import torch.nn as nn
import numpy as np


import random
import json
import argparse
from model.FFTRadNet_noseg import FFTRadNet
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders

from loss import pixor_loss
from dataset.matlab_dataset import MATLAB
from utils.evaluation import run_evaluation

import matplotlib.pyplot as plt

from utils.plots import plot_histograms#, create_video_from_images


import os
import json
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re

# class EpochLossReader:
#     def __init__(self, file_path):
#         self.file_path = file_path

#     def read_epoch_losses(self):
#         epoch_losses = {}
#         with open(self.file_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split(':', 1)
#                 epoch = int(parts[0].split()[1])
#                 loss_str = parts[1]
#                 loss = float(loss_str.split("'loss': ")[1].split(",")[0].strip())
#                 epoch_losses[epoch] = loss
#         return epoch_losses

# class PredictionLoader:
#     def __init__(self, checkpoint_dir):
#         self.checkpoint_dir = checkpoint_dir

#     def load_all_predictions(self):
#         checkpoint_files = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')])
#         predictions = []
#         for checkpoint_file in checkpoint_files:
#             checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
#             checkpoint = torch.load(checkpoint_path)
#             prediction = checkpoint['prediction']  # Assuming prediction is stored in the checkpoint
#             predictions.append(prediction)
#         return predictions

class LossPlotter:
    def __init__(self, output_file, val_division):
        self.output_file = output_file
        self.val_division = val_division

    def plot_validation_losses(self, epoch_train_losses, epoch_val_losses):
        train_epochs = list(range(len(epoch_train_losses)))
        train_losses = epoch_train_losses
        val_epochs = list(range(len(epoch_val_losses)))
        val_losses = epoch_val_losses
        #val_epochs = sorted(epoch_val_losses.keys())
        #val_losses = [epoch_val_losses[epoch] / self.val_division for epoch in val_epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_losses, color='blue', label='Training Loss')
        plt.plot(val_epochs, val_losses, color='green', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 800)
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        max_epoch = max(train_epochs + val_epochs)
        ticks = list(range(0, max_epoch + 1, 10))
        plt.xticks(ticks, rotation=45)
        plt.savefig(self.output_file)
        print(f'Plot saved to {self.output_file}')

class MainProcess:
    def __init__(self, config, checkpoint_dir, histogram, lossplot, plot_file, save_plot_path):
    #def __init__(self, config, checkpoint_dir, histogram, lossplot, output_val_file, plot_file, save_plot_path):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.histogram = histogram
        self.lossplot = lossplot
        #self.output_val_file = output_val_file
        self.plot_file = plot_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.setup_seed()
        self.val_division = None
        self.save_plot_path = save_plot_path

    def setup_seed(self):
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def load_data(self, batch_size):
        enc = ra_encoder(
            geometry=self.config['dataset']['geometry'],
            statistics=self.config['dataset']['statistics'],
            regression_layer=2
        )
        dataset = MATLAB(
            root_dir=self.config['dataset']['root_dir'],
            folder_dir = config['dataset']['data_folder'],
            statistics=self.config['dataset']['statistics'],
            encoder=enc.encode
        )
        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, batch_size, self.config['dataloader'], self.config['seed'])
        self.val_division = len(val_loader.dataset)
        return train_loader, val_loader

    def load_model(self, mimo):
        net = FFTRadNet(
            blocks=self.config['model']['backbone_block'],
            mimo_layer=mimo, #self.config['model']['MIMO_output'],
            Ntx = config['model']['NbTxAntenna'],
            Nrx = config['model']['NbRxAntenna'],
            channels=self.config['model']['channels'],
            regression_layer=2,
            detection_head=self.config['model']['DetectionHead'],
        )
        return net.to('cuda')

    def create_histograms(self, net, dataset, epoch, batch_size):
        all_outputs = []
        for i, data in enumerate(dataset):
            print("read input")
            inputs = data[0].to('cuda').float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs)
                # Collect the outputs
                all_outputs.append(outputs.detach().cpu().numpy().copy())
        # Concatenate all outputs into a single array
        all_outputs = np.concatenate(all_outputs, axis=0)
        # Plot histogram for the entire dataset
        save_path = os.path.join(self.save_plot_path, self.plot_file)
        print("saving all output to", )
        plot_histograms(all_outputs, epoch, self.histogram, batch_size, self.save_plot_path)

    #def calculate_validation_loss(self, net, enc, val_loader):
    #    return run_evaluation(net, val_loader, enc, check_perf=False, detection_loss=pixor_loss, segmentation_loss=None, losses_params=self.config['losses'])

    #def save_epoch_losses(self, epoch_losses_val):
    #    with open(self.output_val_file, 'w') as f:
    #        for epoch, loss in epoch_losses_val.items():
    #            f.write(f'Epoch {epoch}: {loss}\n')

    # def create_video_if_histogram(self):
    #     if self.histogram:
    #         create_video_from_images(self.save_plot_path, fps=2)
    
    def get_latest_checkpoint(self):
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        max_epoch = -1
        latest_checkpoint = None

        for checkpoint_file in checkpoint_files:
            epoch = int(checkpoint_file.split('_')[2][5:])
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = checkpoint_file

        return latest_checkpoint
    def extract_params_from_filename(self, filename):

        epoch_match = re.search(r'_epoch(\d+)', filename)
        mimo_match = re.search(r'_mimo(\d+)', filename)

        epoch = int(epoch_match.group(1)) if epoch_match else None
        mimo_layer = int(mimo_match.group(1)) if mimo_match else None

        return epoch, mimo_layer

    def run(self, batch_size, check_epoch):
        if self.histogram:
            train_loader, val_loader = self.load_data(batch_size)
            
            checkpoint_files = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')])
            for checkpoint_file in checkpoint_files:
                epoch, mimo = self.extract_params_from_filename(checkpoint_file)
                #print("mimo: ", mimo)

                net = self.load_model(mimo)
                
                if check_epoch > 0:
                    if epoch == check_epoch:
                        print("in the right loop")
                        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
                        checkpoint = torch.load(checkpoint_path)
                        net.load_state_dict(checkpoint['net_state_dict'])
                        if self.histogram:
                            dataset = train_loader if self.histogram == "train" else val_loader
                            self.create_histograms(net, dataset, epoch, batch_size)

                else:
                    if epoch % 5 == 0:
                        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
                        checkpoint = torch.load(checkpoint_path)
                        net.load_state_dict(checkpoint['net_state_dict'])
                        if self.histogram:
                            dataset = train_loader if self.histogram == "train" else val_loader
                            self.create_histograms(net, dataset, epoch, batch_size)
            #self.create_video_if_histogram()

        if self.lossplot:
            last_checkpoint = self.get_latest_checkpoint()
            checkpoint_path = os.path.join(self.checkpoint_dir, last_checkpoint)
            dict = torch.load(checkpoint_path)
            history = dict['history']
            epoch_train_losses = history['train_loss']
            epoch_val_losses = history['val_loss']
            LossPlotter(self.plot_file, self.val_division).plot_validation_losses(epoch_train_losses, epoch_val_losses)
            print("Finished loss plotting")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model checking')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpointdir', default=None, type=str, help='Path to the .pth model checkpoint folder to load all checkpoint files')
    parser.add_argument('-his', '--histogram', type=str, help='If provided, process data to create histograms. Provide "train" to indicate train dataset.')
    parser.add_argument('-loss', '--lossplot', action='store_true')
    parser.add_argument('-b', type=int, default=4, help='Batch size')
    parser.add_argument('-e', type=int, help='specific epoch to check')

    args = parser.parse_args()
    print(args)
    config = json.load(open(args.config))
    main_process = MainProcess(config, args.checkpointdir, 
                               args.histogram, 
                               args.lossplot, 
                               '2rx_10000_seqdata', 
                               "/imec/other/dl4ms/chu06/public/plot/FFTRadNet/histogram/")
    main_process.run(args.b, args.e)
