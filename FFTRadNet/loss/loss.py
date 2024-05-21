import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    #def __init__(self, gamma=0, size_average=False):
    def __init__(self, alpha=0.25, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

        self.alpha = alpha

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)

        # compute focal loss
        #loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)
        loss = -self.alpha * (1-pt)**self.gamma * torch.log(pt+1e-6)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
def pixor_loss(batch_predictions, batch_labels,param):


    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0,:, :].contiguous().flatten()
    classification_label = batch_labels[:, 0,:, :].contiguous().flatten()

    if(param['classification']=='FocalLoss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(), classification_label.double(),reduction='sum')
    print('classification_loss')
    print(classification_loss)
    
    #####################
    #  Regression loss  #
    #####################

    regression_prediction = batch_predictions.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0)*
                        regression_prediction.size(1)*regression_prediction.size(2), regression_prediction.size(3)])
    regression_label = batch_labels.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_label = regression_label.contiguous().view([regression_label.size(0)*regression_label.size(1)*
                                                           regression_label.size(2), regression_label.size(3)])

    positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    pos_regression_label = regression_label[positive_mask.squeeze(), :]
    pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]


    T = batch_labels[:,1:]
    P = batch_predictions[:,1:]
    M = batch_labels[:,0].unsqueeze(1)

    batch_predictions[0, 0, :, :]

    # print('regression prediction in batch')
    # print(batch_predictions.shape)
    # print(batch_predictions[0:10, 0])

    # print('regression prediction with onlt positive in batch')
    # print((P*M).shape)
    # print((P*M)[0:10, 0])
    # print('sum P*M', (P*M).sum()) # ex. 2.85
    # print('sum M', M.sum()) # ex. 2.85
    # print('regression labels in batch')
    # print(batch_labels.shape)
    # print(batch_labels[0:10, 1])
    # print('labels sum', batch_labels[:,1:].sum()) #ex. 2014

    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')
    
    regression_loss = reg_loss_fct(P*M,T)
    print('regression_loss')
    print(regression_loss)
    NbPts = M.sum()
    if(NbPts>0):
        regression_loss/=NbPts

    return classification_loss,regression_loss
    



    
    