import torch
import numpy as np
from .metrics import GetFullMetrics, Metrics
import pkbar

import matplotlib.pyplot as plt
import os
import torch.nn as nn
from utils.plots import matrix_plot, detection_plot
import time
from collections import Counter

def run_evaluation(trial, net,loader,encoder,threshold, check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    #encoder_threshold = trial.suggest_float("encoder_threshold", 0.01, 0.07, step=0.02) # paper set 0.05
    #threshold = trial.suggest_float("FFT_confidence_threshold", 0.1, 0.2, step=0.05) # paper set 0.2
    start_time = time.time()
    for i, data in enumerate(loader):
        

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs, label_map,losses_params)           
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]

            loss = classif_loss + reg_loss 
            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs.detach().cpu().numpy().copy()
            labels = data[2] # data[2]=labels(=box_labels)


            for pred_obj,true_obj in zip(out_obj,labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map

                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=5,range_max=100) 
                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=0,range_max=345) 
                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                             threshold=threshold,range_min=0,range_max=345) 
                

        # Record the end time and calculate the computation time for this iteration.

        #computation_times.append(computation_time)
        
                


        kbar.update(i)
        
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Iteration {i}: Computation time = {computation_time:.4f} seconds")
    mAP,mAR, mIoU = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}
    #return {'loss':running_loss}

def run_evaluation_(net,loader,encoder,threshold, check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    start_time = time.time()

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()
        #seg_map_label = data[2].to('cuda').double()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs, label_map,losses_params)           

            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            #loss_seg *=losses_params['weight'][2]


            loss = classif_loss + reg_loss #+ loss_seg

            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs.detach().cpu().numpy().copy()
            labels = data[2]#data[3] # data[3]=labels(=box_labels)

            #out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            #label_freespace = seg_map_label.detach().cpu().numpy().copy()

            #for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels,label_freespace):
            for pred_obj,true_obj in zip(out_obj,labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map
                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                            threshold=threshold,range_min=0,range_max=345) 

                
        # Record the end time and calculate the computation time for this iteration.
        end_time = time.time()
        computation_time = end_time - start_time
        #computation_times.append(computation_time)
        print(f"Iteration {i}: Computation time = {computation_time:.4f} seconds")

        kbar.update(i)
        

    mAP,mAR, mIoU = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}

def run_evaluation_ddp(net, loader,encoder, gpu, optuna_config, check_perf=False, detection_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    

    #encoder_threshold = trial.suggest_float("encoder_threshold", 0.01, 0.07, step=0.02) # paper set 0.05
    #FFT_confidence_threshold = trial.suggest_float("FFT_confidence_threshold", 0.1, 0.5, step=0.1) # paper set 0.2
    # Initialize a list to store the computation times for each iteration.
    computation_times = []

    for i, data in enumerate(loader):
        # Record the start time.
        start_time = time.time()

        # input, out_label,segmap,labels
        inputs = data[0].to(gpu).float()
        label_map = data[1].to(gpu).float()


        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs, label_map,losses_params)           
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]

            loss = classif_loss + reg_loss 
            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs.detach().cpu().numpy().copy()
            labels = data[2] # data[2]=labels(=box_labels)


            for pred_obj,true_obj in zip(out_obj,labels):
                pred_map = 0 # not used in update() for computing pred_map=PredMap
                true_map = 0 # not used in update() for computing ture_map=lable_map

                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=5,range_max=100) 
                #metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                #            threshold=0.2,range_min=0,range_max=345) 
                metrics.update(pred_map,true_map,np.asarray(encoder.decode(pred_obj,optuna_config['threshold'])),true_obj,
                             threshold=optuna_config['threshold'],range_min=0,range_max=345) 
                

        # Record the end time and calculate the computation time for this iteration.
        end_time = time.time()
        computation_time = end_time - start_time
        #computation_times.append(computation_time)
        print(f"Iteration {i}: Computation time = {computation_time:.4f} seconds")


        

    mAP,mAR, mIoU, TP, FP, FN = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU, 'TP': TP, 'FP':FP, 'FN':FN}
    #return {'loss':running_loss}

def run_FullEvaluation(net,loader,encoder,iou_threshold=0.5):

    net.eval()
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):
        
        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        
        labels_object = data[3]
        label_freespace = data[2].numpy().copy()
            
        for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
            
            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

            predictions['prediction']['freespace'].append(pred_map[0])
            predictions['label']['freespace'].append(true_map)



        kbar.update(i)
        
    #GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=0.5)
    GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=345,IOU_threshold=0.5)


    # doesn't need freespace evaluation
    # mIoU = []
    # for i in range(len(predictions['prediction']['freespace'])):
    #     # 0 to 124 means 0 to 50m
    #     pred = predictions['prediction']['freespace'][i][:124].reshape(-1)>=0.5
    #     label = predictions['label']['freespace'][i][:124].reshape(-1)
        
    #     intersection = np.abs(pred*label).sum()
    #     union = np.sum(label) + np.sum(pred) -intersection
    #     iou = intersection /union
    #     mIoU.append(iou)


    # mIoU = np.asarray(mIoU).mean()
    # print('------- Freespace Scores ------------')
    # print('  mIoU',mIoU*100,'%')

def run_FullEvaluation_(net,loader,encoder,iou_threshold=0.5):

    net.eval()
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):
        
        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs.detach().cpu().numpy().copy()
 
        labels_object = data[2]
            
        for pred_obj,true_obj in zip(out_obj,labels_object):
            
            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

        kbar.update(i)
        
    #GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=0.5)
    map, mar, f1_score, mRange, mAngle = GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=345,IOU_threshold=0.5)

    return map, mar, f1_score, mRange, mAngle

def run_iEvaluation(base_dir, net,loader,encoder,epochs, trial_num, datamode,iou_threshold=0.5):

    net.eval()
    if datamode == 'train':
        division = 1
    else:
        division = 1


    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    targetnum_list = []
    for i, data in enumerate(loader):


        if (i % division == 0): #len(loader) - 1:
                    # input, out_label,segmap,labels
            inputs = data[0].to('cuda').float()


            with torch.set_grad_enabled(False):
                outputs = net(inputs)

            label_map = data[1].to('cuda').float() # debugging

            out_obj = outputs.detach().cpu().numpy().copy()
            #out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            
            matrix_plot(base_dir, outputs, label_map, trial_num, datamode, epoch=epochs, batch = i) # debugging  

            num_target = detection_plot(base_dir, outputs, label_map, trial_num, datamode, epoch=epochs, batch = i)   
            targetnum_list.extend(num_target)
            #print(num_target)
            #print(targetnum_list)
            # labels_object = data[3]
            # label_freespace = data[2].numpy().copy()
            # print("out_obj : ", out_obj.shape)
                
            # for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
                
            #     print("pred_obj : ", pred_obj.shape)
            #     print("true_obj : ", true_obj.shape)
            #     print("np.asarray(encoder.decode(pred_obj,0.05)) : ", np.asarray(encoder.decode(pred_obj,0.05)).shape)
            #     predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            #     predictions['label']['objects'].append(true_obj)

            #     predictions['prediction']['freespace'].append(pred_map[0])
            #     predictions['label']['freespace'].append(true_map)
            
            # print("predictions['prediction']['objects'] : ", len(predictions['prediction']['objects']))
            # print("predictions['label']['objects'] : ", len(predictions['label']['objects']))
            #GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=345,IOU_threshold=0.5)
        kbar.update(i)

    # Count the occurrences of each element
    count = Counter(targetnum_list)
    print("after counter")
    #print(count)
    
    stat_file = os.path.join(base_dir, 'target_counts.txt')
    with open(stat_file, 'a') as f:
    # Print the counts
        for number, frequency in count.items():
            f.write(f"datamode: {datamode}\n")
            f.write(f"Number {number} occurs {frequency} times\n")

        

