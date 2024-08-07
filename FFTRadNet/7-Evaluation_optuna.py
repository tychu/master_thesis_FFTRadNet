import os
import json
import argparse
import torch
import random
import numpy as np
from model.FFTRadNet_ddp import FFTRadNet
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.nn.functional as F

from utils.evaluation import run_evaluation_, run_evaluation, run_iEvaluation # without trial
import torch.nn as nn
from loss import pixor_loss

from dataset.matlab_dataset import MATLAB

from pathlib import Path
from datetime import datetime
import re

def extract_params_from_filename(filename):
    trial_match = re.search(r'_trialnumber_(\d+)', filename)
    batch_match = re.search(r'_batch(\d+)', filename)
    mimo_match = re.search(r'_mimo(\d+)', filename)
    print(batch_match)
    print(mimo_match)
    print(trial_match)

    trial_num = int(trial_match.group(1)) if trial_match else None
    batch_size = int(batch_match.group(1)) if batch_match else None
    mimo_layer = int(mimo_match.group(1)) if mimo_match else None
    print(trial_num)
    print(batch_size)
    print(mimo_layer)

    return trial_num, batch_size, mimo_layer



def evaluate_checkpoint(config, base_dir, checkpoint, batch_size, mimo_layer, trial_num, check_epoch):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry=config['dataset']['geometry'], 
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)
    
    dataset = MATLAB(root_dir=config['dataset']['root_dir'],
                     folder_dir = config['dataset']['data_folder'],
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, batch_size, config['dataloader'], config['seed'])

    # Create the model
    net = FFTRadNet(blocks=config['model']['backbone_block'],
                    mimo_layer= 64, #mimo_layer, #config['model']['MIMO_output'],
                    Ntx = config['model']['NbTxAntenna'],
                    Nrx = config['model']['NbRxAntenna'],
                    channels=config['model']['channels'], 
                    regression_layer=2, 
                    detection_head=config['model']['DetectionHead'])
    
    print("Parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
    net.to('cuda')

    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint)
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')
    

    epoch=10
    thresholds = [0.2]
    for threshold in thresholds:

        eval = run_evaluation_(net, val_loader, enc, threshold, check_perf=(epoch >= 1),
                                detection_loss=pixor_loss, segmentation_loss=None,
                                losses_params=config['losses']) # only run some plot to check the output

        tra = run_evaluation_(net, train_loader, enc, threshold, check_perf=(epoch >= 1),
                                detection_loss=pixor_loss, segmentation_loss=None,
                                losses_params=config['losses']) # only run some plot to check the output

        #run_iEvaluation(base_dir, net,train_loader,enc, check_epoch, trial_num, datamode='train')
        #run_iEvaluation(base_dir, net,val_loader,enc, check_epoch, trial_num, datamode='val')

        if eval['mAP'] + eval['mAR'] == 0:
            F1_score = 0
        else:
            F1_score = (eval['mAP']*eval['mAR'])/((eval['mAP'] + eval['mAR'])/2)
            print("eval F1: ", F1_score)
        
        if tra['mAP'] + tra['mAR'] == 0:
            F1_score_tra = 0
        else:
            F1_score_tra = (tra['mAP']*tra['mAR'])/((tra['mAP'] + tra['mAR'])/2)
            print("train F1: ", F1_score_tra)

        # save the score of F1, mAP, mAR
        
        stat_file = os.path.join(base_dir, 'evaluation_scores_valshuffle.txt')
        with open(stat_file, 'a') as f:
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Targets threshold: {threshold}\n")
            f.write(f"Batch size: {batch_size}, MIMO layer: {mimo_layer}\n")
            f.write("validation\n")
            f.write(f"precision: {eval['mAP']}, recall: {eval['mAR']}, F1 Score: {F1_score}\n")
            f.write("train\n")
            f.write(f"precision: {tra['mAP']}, recall: {tra['mAR']}, F1 Score: {F1_score_tra}\n")
            f.write("\n")

def main(config, base_dir, check_epoch):
    # base_dir: with multiple checkpoint folder
    batch_size = 0
    mimo_layer = 0
    for root, dirs, files in os.walk(base_dir):
        if root != base_dir:
            print("root: ", root, "dirs:", dirs)
            #max_epoch = -1
        
            for file in files:
                if file.endswith('.pth'):
                    epoch = int(file.split('_')[2][5:])
                    if epoch == check_epoch:
                        #max_epoch = epoch
                        latest_checkpoint = file
                        
                        checkpoint_path = os.path.join(root, latest_checkpoint)
                        print(checkpoint_path)
                        trial_num, batch_size, mimo_layer = extract_params_from_filename(checkpoint_path)
            if batch_size > 0 and mimo_layer > 0:
                print(f"Evaluating checkpoint: {checkpoint_path}")
                print(f"Batch size: {batch_size}, MIMO layer: {mimo_layer}")
                evaluate_checkpoint(config, base_dir, checkpoint_path, batch_size, mimo_layer, trial_num, check_epoch)




if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-d', '--checkpoint_dir', required=True, type=str,
                        help='Directory containing checkpoint files')
    parser.add_argument('-e', '--epoch', required=True, type=int,
                        help='the sprcific epoch to check')

    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint_dir, args.epoch)


