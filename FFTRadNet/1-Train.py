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

def main(config, resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    print(exp_name)

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize tensorboard
    #writer = SummaryWriter(output_folder / exp_name)

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)
    
    #dataset = RADIal(root_dir = config['dataset']['root_dir'],
    #                    statistics= config['dataset']['statistics'],
    #                    encoder=enc.encode,
    #                    difficult=True)
    
    dataset = MATLAB(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])
    


    # Create the model
    net = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = config['model']['MIMO_output'],
                        channels = config['model']['channels'], 
                        regression_layer = 2, 
                        detection_head = config['model']['DetectionHead'], 
                        segmentation_head = config['model']['SegmentationHead'])

    net.to('cuda')


    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
    best_mAP = 0

    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')


    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)


    for epoch in range(startEpoch,num_epochs):
        
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        
        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):


            inputs = data[0].to('cuda').float()
            # debugging
            if i == 0:
                print(inputs.size())


            label_map = data[1].to('cuda').float()
            if(config['model']['SegmentationHead']=='True'):
                seg_map_label = data[2].to('cuda').double()

            # reset the gradient
            optimizer.zero_grad()
            
            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)


            classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map,config['losses'])           
               
            #prediction = outputs['Segmentation'].contiguous().flatten()
            #label = seg_map_label.contiguous().flatten()        
            #loss_seg = freespace_loss(prediction, label)
            #loss_seg *= inputs.size(0)

            classif_loss *= config['losses']['weight'][0]
            reg_loss *= config['losses']['weight'][1]
            #loss_seg *=config['losses']['weight'][2]


            loss = classif_loss + reg_loss #+ loss_seg

            #writer.add_scalar('Loss/train', loss.item(), global_step)
            #writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            #writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
            #writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)

            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
        
            #kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), #("reg", reg_loss.item()),("freeSpace", loss_seg.item())])
            
            kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()) ] )
            
            global_step += 1

            # # Check if this is the last iteration
            # if (i == 50 and epoch > 2 and epoch % 10 == 0):
            # #if (i == len(train_loader) - 1 and epoch != 0):
            #     print(f"Last iteration in epoch {epoch}: batch {i}")
            #     print("let's plot!!!!!!!!!!!!!!!!!!!!!!!!")
            #     # plot the prediction and ground truth, pixel occupied with vehicle (RA coordinate) 
            #     #detection_plot(outputs['Detection'], label_map, epoch)
            #     matrix_plot(outputs['Detection'], label_map, epoch)

            # if (i == 50 and epoch % 10 == 0):
            #     outputs_to_save = outputs['Detection'].detach().cpu().numpy().copy()
            #     labels_to_save = label_map.detach().cpu().numpy().copy()
            #     inputs_to_save = inputs.detach().cpu().numpy().copy()
            #     save_path = os.path.join(config['dataset']['root_dir'], 'output_detection/', f'output_detection_{epoch}')
            #     np.savez(save_path, output = outputs_to_save, labels = labels_to_save, input = inputs_to_save)
            #     print(f'Slice saved to {save_path}')

                


        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        
        ######################
        ## validation phase ##
        ######################

        #eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=10),
        #                        detection_loss=pixor_loss,segmentation_loss=freespace_loss,
        #                        losses_params=config['losses'])
        eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=10),
                                detection_loss=pixor_loss,segmentation_loss=None,
                                losses_params=config['losses'])

        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])
        #history['mIoU'].append(eval['mIoU'])

        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])
        #kbar.add(1, values=[("val_loss", eval['loss'])])

            

        #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        #writer.add_scalar('Loss/test', eval['loss'], global_step)
        #writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
        #writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
        #writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

        # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
        #name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])
        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}.pth'.format(epoch, loss, eval['mAP'], eval['mAR'])
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        # plot the loss
        if (epoch % 10 == 0 and epoch != 0):
            loss_plot(history, epoch)

        torch.save(checkpoint,filename)
          
        print('')
        # if (epoch > 2 and epoch % 10 == 0):
        #     for i, data in enumerate(train_loader):
        #         if i == 5:
        #             inputs = data[0].to('cuda').float()
        #             label_map = data[1].to('cuda').float()
        #             with torch.set_grad_enabled(False):
        #                 outputs = net(inputs)

        #             print(f"Last iteration in epoch {epoch}: batch {i}")
        #             print("let's plot!!!!!!!!!!!!!!!!!!!!!!!!")
        #                 # plot the prediction and ground truth, pixel occupied with vehicle (RA coordinate) 
        #                 #detection_plot(outputs['Detection'], label_map, epoch)
        #             matrix_plot(outputs['Detection'], label_map, epoch)




# check input        
def rd_plot(data): 
    directory = './plot/'
    data_ = 20* np.log10(np.abs(data.detach().cpu().numpy().copy()))
    

    plt.figure(figsize=(6, 6))
    plt.imshow(data_)
    # Save the plot with an incrementally named file
    filepath = os.path.join(directory, f'RDplot.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close() 
        
# ### plot detection (classification)
# def detection_plot(predictions, labels, epoch):
#     prediction = predictions[:, 0, :, :]
#     print(prediction[0, 0:10, 0])
#     #target_prediction = (prediction > 0.5).float()
#     label = labels[:, 0, :, :]
#     # Specify the directory to save the plot
#     directory = './plot/'
#     # Iterate through each matrix
#     print("plotting data shape")
#     print(prediction.shape)
#     target_num = 0
#     for m in range(prediction.shape[0]):
#         # Extract the current matrix
#         pre = prediction[m]
#         lab = label[m]

#         # Create a figure
#         plt.figure(figsize=(6, 6))

#         # Plot pre: Red points
#         for i in range(pre.shape[0]):
#             for j in range(pre.shape[1]):
#                 #if pre[i, j] == 1:
#                 if pre[i, j] > 0.2:
#                     target_num += 1
#                     #print("predict target!!!")
#                     plt.scatter(j, i, color='red', s=1, label='prediction' if i == 0 and j == 0 else "")
#         print('number of targets in the prediction', target_num)
#         # Plot lab: Blue points
#         for i in range(lab.shape[0]):
#             for j in range(lab.shape[1]):
#                 if lab[i, j] == 1:
#                     plt.scatter(j, i, color='blue', s=1, label='ground truth' if i == 0 and j == 0 else "")

#         # Set plot limits and labels
#         plt.xlim(0, pre.shape[1])
#         plt.ylim(0, pre.shape[0])
#         #plt.gca().invert_yaxis()  # To match matrix indexing
#         plt.xlabel('angle Index')
#         plt.ylabel('range Index')
#         plt.title('Comparison of prediction and labels')
        
#         # Save the plot with an incrementally named file
#         filepath = os.path.join(directory, f'plot_{epoch}_{m}.png')
#         plt.savefig(filepath)
#         print(f'Plot saved to {filepath}')

#         # Close the plot to free up memory
#         plt.close()        


# def matrix_plot(predictions, labels, epoch):
#     directory = './plot_0612_16rx_detreg/'
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     prediction = predictions[0, 0, :, :].detach().cpu().numpy().copy()
#     #target_prediction = (prediction > 0.5).float()
#     label = labels[0, 0, :, :].detach().cpu().numpy().copy()

#     m1 = axs[0].imshow(prediction, cmap='magma', interpolation='none')
#     axs[0].set_title('prediction')
#     axs[0].set_ylim(0, prediction.shape[0])
#     axs[0].set_xlim(0, prediction.shape[1])
#     axs[0].set_xlabel('azimuth')
#     axs[0].set_ylabel('range')

#     fig.colorbar(m1, ax=axs[0])

#     # Plot the second matrix
#     m2 = axs[1].imshow(label, cmap='magma', interpolation='none', vmin=0.0, vmax=1.0)
#     axs[1].set_title('label')
#     axs[1].set_ylim(0, label.shape[0])
#     axs[1].set_xlim(0, label.shape[1])
#     axs[1].set_xlabel('azimuth')
#     axs[1].set_ylabel('range')

#     fig.colorbar(m2, ax=axs[1])

#     # Save the plot with an incrementally named file
#     filepath = os.path.join(directory, f'matrix_plot_{epoch}.png')
#     plt.savefig(filepath)
#     print(f'Plot saved to {filepath}')

#     # Close the plot to free up memory
#     plt.close()    

def loss_plot(history, epoch):
    directory = './plot_0618_16rx_detreg_notrunningstat_BNlayer/'
    # Plot the training loss curve
    plt.figure()
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    # Save the plot
    filepath = os.path.join(directory, f'Train_loss_curve_{epoch}.png')
    plt.savefig(filepath)
    plt.close()

    # Plot the validating loss curve
    plt.figure()
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    # Save the plot
    filepath = os.path.join(directory, f'Validation_loss_curve_{epoch}.png')
    plt.savefig(filepath)
    plt.close()

    print(f"Loss curve saved to {filepath}")

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')

    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.resume)
