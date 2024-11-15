import os
import json
import argparse
import torch
import random
import numpy as np
from model.FFTRadNet_noseg import FFTRadNet
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.nn.functional as F

from utils.evaluation import run_evaluation_, run_FullEvaluation_, run_evaluation, run_iEvaluation # without trial
import torch.nn as nn
from loss import pixor_loss

from dataset.matlab_dataset_ddp import MATLAB

import io

def main(config, checkpoint,difficult):

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
        
    batch_size = 4
    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, batch_size, config['dataloader'],config['seed'])

    _mimo_layer = 128 #mimo_layer if mimo_layer  else 128
    # Create the model
    net = FFTRadNet(blocks=config['model']['backbone_block'],
                    mimo_layer= _mimo_layer, #config['model']['MIMO_output'],
                    Ntx = config['model']['NbTxAntenna'],
                    Nrx = config['model']['NbRxAntenna'],
                    channels=config['model']['channels'], 
                    regression_layer=2, 
                    detection_head=config['model']['DetectionHead'])
    
    print("Parameters: ",sum(p.numel() for p in net.parameters() if p.requires_grad))

    net.to('cuda')


    print('===========  Loading the model ==================:')

    dict = torch.load(checkpoint)
    net.load_state_dict(dict['net_state_dict'])

    print('===========  Running the evaluation ==================:')
    output_file = os.path.join('FFTRadNet_detection_score_optimal.txt')
    print("Saving scores to:", output_file)
    with open(output_file, 'a') as f:
        f.write('------- Train ------------\n')
    map_train, mar_train, f1_score_train, mRange_train, mAngle_train = run_FullEvaluation_(net,train_loader,enc) #test_loader
    with open(output_file, 'a') as f:
        f.write('------- Validation ------------\n')
    map_val, mar_val, f1_score_val, mRange_val, mAngle_val = run_FullEvaluation_(net,val_loader,enc) 
    with open(output_file, 'a') as f:
        f.write('------- Test ------------\n')
    map_test, mar_test, f1_score_test, mRange_test, mAngle_test = run_FullEvaluation_(net,test_loader,enc) 

    # Transpose the data, excluding the IOU_threshold (index 0)
    #transposed_train_data = list(zip(*result_train))[1:]  # Skips the first element (IOU_threshold)
    #ransposed_val_data = list(zip(*result_val))[1:]  # Skips the first element (IOU_threshold)
    #ransposed_test_data = list(zip(*result_test))[1:]  # Skips the first element (IOU_threshold)

    # Calculate the mean of each metric
    #means_train = [np.mean(metric) for metric in transposed_train_data]
    #means_val = [np.mean(metric) for metric in ransposed_val_data]
    #means_test = [np.mean(metric) for metric in ransposed_test_data]
    
    with open(output_file, 'a') as f:
        f.write('------- Summary ------------\n')
        f.write('------- Train ------------\n')
        f.write('Means of Precision, Recall, F1_score, RangeError, AngleError:\n')
        f.write(f"{map_train, mar_train, f1_score_train, mRange_train, mAngle_train}\n")
        f.write('------- Validation ------------\n')
        f.write('Means of Precision, Recall, F1_score, RangeError, AngleError:\n')
        f.write(f"{map_val, mar_val, f1_score_val, mRange_val, mAngle_val}\n")
        f.write('------- Test ------------\n')
        f.write('Means of Precision, Recall, F1_score, RangeError, AngleError:\n')
        f.write(f"{ map_test, mar_test, f1_score_test, mRange_test, mAngle_test}\n")




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
