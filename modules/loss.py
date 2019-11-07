import torch 
from torch import nn 


class SiameseLoss(nn.Module):
    """Compute Siamese loss of paper: 
    Learning Text Similarity with Siamese Recurrent Networks
    Used only for binary similarity classification
    """
    def __init__(self, threshold):
        super(SiameseLoss, self).__init__()
        self.threshold = threshold
    
    
    def forward(self, pred, targets):
        # pred: batch_size, 1
        # targets: batch_size
        pos_condition = (targets == 1)
        l_pos = pred[pos_condition]
        l_pos_loss = (1/4*(1-l_pos)**2).sum()
        
        neg_condition = (targets == 0)
        l_neg = pred[neg_condition]
        l_neg
        l_neg_loss = (l_neg**2).masked_select(l_neg>self.threshold).sum()

        total_loss = l_pos_loss + l_neg_loss 
        return total_loss