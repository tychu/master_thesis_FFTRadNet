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

from pathlib import Path
from datetime import datetime
import re
import sys
import tarfile

def extract_and_load_model(root, file):
    """
    Extracts a .tar.gz file and loads the PyTorch model from the extracted .pth file.

    :param tar_gz_path: Path to the .tar.gz file.

    :return: The loaded model parameters or model instance if model_class is provided.
    """
    tar_gz_path = os.path.join(root, file)
    try:
        # Step 1: Extract the .tar.gz file
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            tar.extractall(path=root)
            print(f"Extracted {tar_gz_path} to {root}")

        # Step 2: Find the extracted .pth file
        extracted_files = os.listdir(root)
        pth_files = [f for f in extracted_files if f.endswith('.pth')]
        
        if not pth_files:
            raise FileNotFoundError("No .pth file found in the extracted files.")
        
        pth_file_path = os.path.join(root, pth_files[0])
        print(f"Found .pth file: {pth_file_path}")


        return pth_files[0]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
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

    # if the checkpoint file do not have mimo in file name
    _mimo_layer = mimo_layer if mimo_layer  else 128
    print("_mimo_layer: ", _mimo_layer)
    
    # Create the model
    net = FFTRadNet(blocks=config['model']['backbone_block'],
                    mimo_layer= _mimo_layer, #config['model']['MIMO_output'],
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
    
    #run_iEvaluation(base_dir, net,train_loader,enc, check_epoch, trial_num, datamode='train')

    #run_iEvaluation(base_dir, net,val_loader,enc, check_epoch, trial_num, datamode='val')
    
    map_val, mar_val, f1_score_val, mRange_val, mAngle_val = run_FullEvaluation_(net,val_loader,enc)
    map_train, mar_train, f1_score_train, mRange_train, mAngle_train = run_FullEvaluation_(net,train_loader,enc)
    map_test, mar_test, f1_score_test, mRange_test, mAngle_test = run_FullEvaluation_(net,test_loader,enc)

    output_file = os.path.join(base_dir, 'detection_score.txt')
    print("Saving scores to:", output_file)
    with open(output_file, 'a') as f:
        f.write('------- Validation ------------\n')
        f.write('  mAP: {0}\n'.format(map_val))
        f.write('  mAR: {0}\n'.format(mar_val))
        f.write('  F1 score: {0}\n'.format(f1_score_val))
        f.write('------- Regression Errors------------\n')
        f.write('  Range Error:: {0}\n'.format(mRange_val))
        f.write('  Angle Error: {0}\n'.format(mAngle_val))        
        f.write('------- Train ------------\n')    
        f.write('  mAP: {0}\n'.format(map_train))
        f.write('  mAR: {0}\n'.format(mar_train))
        f.write('  F1 score: {0}\n'.format(f1_score_train))
        f.write('------- Regression Errors------------\n')
        f.write('  Range Error:: {0}\n'.format(mRange_train))
        f.write('  Angle Error: {0}\n'.format(mAngle_train))   
        f.write('------- Test ------------\n')    
        f.write('  mAP: {0}\n'.format(map_test))
        f.write('  mAR: {0}\n'.format(mar_test))
        f.write('  F1 score: {0}\n'.format(f1_score_test))
        f.write('------- Regression Errors------------\n')
        f.write('  Range Error:: {0}\n'.format(mRange_test))
        f.write('  Angle Error: {0}\n'.format(mAngle_test))   
    

    thresholds = [0.2]
    for threshold in thresholds:

        eval = run_evaluation_(net, val_loader, enc, threshold, check_perf=True,
                                detection_loss=pixor_loss, segmentation_loss=None,
                                losses_params=config['losses']) # only run some plot to check the output

        tra = run_evaluation_(net, train_loader, enc, threshold, check_perf=True,
                                detection_loss=pixor_loss, segmentation_loss=None,
                                losses_params=config['losses']) # only run some plot to check the output

        test = run_evaluation_(net, test_loader, enc, threshold, check_perf=True,
                                detection_loss=pixor_loss, segmentation_loss=None,
                                losses_params=config['losses']) # only run some plot to check the output    
    



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

        if test['mAP'] + test['mAR'] == 0:
            F1_score_test = 0
        else:
            F1_score_test = (test['mAP']*test['mAR'])/((test['mAP'] + test['mAR'])/2)
            print("train F1: ", F1_score_test)
        # save the score of F1, mAP, mAR
        
        stat_file = os.path.join(base_dir, 'evaluation_scores_10000samples.txt')
        with open(stat_file, 'a') as f:
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Targets threshold: {threshold}\n")
            f.write(f"Batch size: {batch_size}, MIMO layer: {mimo_layer}\n")
            f.write("train\n")
            f.write(f"precision: {tra['mAP']}, recall: {tra['mAR']}, F1 Score: {F1_score_tra}\n")
            f.write("validation\n")
            f.write(f"precision: {eval['mAP']}, recall: {eval['mAR']}, F1 Score: {F1_score}\n")
            f.write("Test\n")
            f.write(f"precision: {test['mAP']}, recall: {test['mAR']}, F1 Score: {F1_score_test}\n")
            f.write("\n")

def main(config, base_dir, check_epoch):
    # base_dir: with multiple checkpoint folder
    batch_size = 0
    mimo_layer = 0
    print("base_dir: ", base_dir)
    for root, dirs, files in os.walk(base_dir):
        print("root: ", root, "dirs:", dirs)
        if base_dir: #root != base_dir:
            print("root: ", root, "dirs:", dirs)
            #max_epoch = -1
        
            for file in files:
                if file.endswith('.pth') or file.endswith('.gz'):
                    epoch = int(file.split('_')[2][5:])
                    if epoch == check_epoch:
                        #max_epoch = epoch
                        if file.endswith('.gz'):
                            
                            latest_checkpoint = extract_and_load_model(root, file)
                            print("latest_checkpoint: ", latest_checkpoint)
                            
                        else:
                            latest_checkpoint = file
                            print("latest_checkpoint: ", latest_checkpoint)
                        
                        checkpoint_path = os.path.join(root, latest_checkpoint)
                        print(checkpoint_path)
                        trial_num, batch_size, mimo_layer = extract_params_from_filename(checkpoint_path)
            if batch_size > 0 or mimo_layer > 0:
                print(f"Evaluating checkpoint: {checkpoint_path}")
                print(f"Batch size: {batch_size}, MIMO layer: {mimo_layer}")
                evaluate_checkpoint(config, base_dir, checkpoint_path, batch_size, mimo_layer, trial_num, check_epoch)
                break # the script run twice, one for tar.gz one for pth




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
    print(args)

    config = json.load(open(args.config))
    
    main(config, args.checkpoint_dir, args.epoch)


