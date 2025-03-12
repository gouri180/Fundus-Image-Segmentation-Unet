##loss function##


import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice_BCE(nn.Module):
    def __init__(self, weights=None, size_avg=True):
        super(Dice_BCE, self).__init__()
        self.weights = weights
        self.size_avg = size_avg
        
    def forward(self, output_img, label_img, alpha=1):
        output_img = torch.sigmoid(output_img)
        
        if self.weights is not None:
            output_img = output_img * self.weights
        
        output_img = output_img.view(-1)
        labels = label_img.view(-1)
        
        prediction = (output_img*labels).sum()
        dice_loss = 1 - (2*prediction+alpha)/(output_img.sum()+labels.sum()+alpha)
        
        BCE = F.binary_cross_entropy(output_img, labels, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE       
        