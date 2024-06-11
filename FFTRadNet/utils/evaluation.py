import torch
import numpy as np
from .metrics import GetFullMetrics, Metrics
import pkbar

import matplotlib.pyplot as plt
import os

def run_evaluation(net,loader,encoder,check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()
        seg_map_label = data[2].to('cuda').double()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        #if(detection_loss!=None and segmentation_loss!=None):
        if(detection_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs['Detection'], label_map,losses_params)           
            #prediction = outputs['Segmentation'].contiguous().flatten()
            #label = seg_map_label.contiguous().flatten()        
            #loss_seg = segmentation_loss(prediction, label)
            #loss_seg *= inputs.size(0)
                

            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            #loss_seg *=losses_params['weight'][2]


            loss = classif_loss + reg_loss #+ loss_seg

            # statistics
            running_loss += loss.item() * inputs.size(0)

        # if(check_perf):
        #     out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        #     labels = data[3] # data[3]=labels(=box_labels)

        #     out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        #     label_freespace = seg_map_label.detach().cpu().numpy().copy()

        #     for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels,label_freespace):

        #         metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
        #                     threshold=0.2,range_min=5,range_max=100) 
                


        kbar.update(i)
        

    #mAP,mAR, mIoU = metrics.GetMetrics()

    #return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}
    return {'loss':running_loss}


def run_FullEvaluation(net,loader,encoder,iou_threshold=0.5):

    net.eval()
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):

        # # input, out_label,segmap,labels
        # inputs = data[0].to('cuda').float()

        # with torch.set_grad_enabled(False):
        #     outputs = net(inputs)

        # out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        # out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        
        # labels_object = data[3] # box_labels = pd.read_csv(csv_file).to_numpy()  
        #                         # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        #                         # box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 
        # label_freespace = data[2].numpy().copy()
            
        # for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
            
        #     predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
        #     predictions['label']['objects'].append(true_obj)

        #     predictions['prediction']['freespace'].append(pred_map[0])
        #     predictions['label']['freespace'].append(true_map)
        
        if i == 50: #len(loader) - 1:
                    # input, out_label,segmap,labels
            inputs = data[0].to('cuda').float()

            with torch.set_grad_enabled(False):
                outputs = net(inputs)

            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            
            labels_object = data[3] # box_labels = pd.read_csv(csv_file).to_numpy()  
                                    # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
                                    # box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 
            label_freespace = data[2].numpy().copy()
                
            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
                
                predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
                predictions['label']['objects'].append(true_obj)

                predictions['prediction']['freespace'].append(pred_map[0])
                predictions['label']['freespace'].append(true_map)

            # debugging        
            GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=350,IOU_threshold=0.5)#range_min=5,range_max=100

            label_map = data[1].to('cuda').float() # debugging
            matrix_plot(outputs['Detection'], label_map) # debugging

            break


        kbar.update(i)


        
        
    # GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=0,range_max=350,IOU_threshold=0.5)#range_min=5,range_max=100    


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



def matrix_plot(predictions, labels):
    directory = './plot/'
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    prediction = predictions[0, 0, :, :].detach().cpu().numpy().copy()
    #target_prediction = (prediction > 0.5).float()
    label = labels[0, 0, :, :].detach().cpu().numpy().copy()

    m1 = axs[0].imshow(prediction, cmap='magma', interpolation='none')
    axs[0].set_title('prediction')
    axs[0].set_ylim(0, prediction.shape[0])
    axs[0].set_xlim(0, prediction.shape[1])
    axs[0].set_xlabel('azimuth')
    axs[0].set_ylabel('range')

    fig.colorbar(m1, ax=axs[0])

    # Plot the second matrix
    m2 = axs[1].imshow(label, cmap='magma', interpolation='none', vmin=0.0, vmax=1.0)
    axs[1].set_title('label')
    axs[1].set_ylim(0, label.shape[0])
    axs[1].set_xlim(0, label.shape[1])
    axs[1].set_xlabel('azimuth')
    axs[1].set_ylabel('range')

    fig.colorbar(m2, ax=axs[1])

    # Save the plot with an incrementally named file
    filepath = os.path.join(directory, f'matrix_plot_last.png')
    plt.savefig(filepath)
    print(f'Plot saved to {filepath}')

    # Close the plot to free up memory
    plt.close()        