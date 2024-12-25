import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    
    cls_num_list = np.array(cls_num_list)
    per_cls_weights = (1. - beta)/(1e-4 + 1. - beta**cls_num_list)
    per_cls_weights = len(cls_num_list)*per_cls_weights/np.sum(per_cls_weights)

    print(per_cls_weights)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        
        N, num_classes = input.shape
        target_id = np.array(target, dtype=np.int32)

        celoss = nn.CrossEntropyLoss(reduce=False)
        logprobs = -1*celoss(input, target)
        focal_loss = torch.pow((1 - torch.exp(logprobs)), self.gamma)*logprobs

        weights = torch.tensor(self.weight[target_id])
        loss = -torch.sum(weights*focal_loss)/N

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
