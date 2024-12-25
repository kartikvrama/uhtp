import torch
from torch import nn

class SkeletonClassifier(nn.Module):
    """
    the neural network classifier for skeletal joint data
    """
    def __init__(self, num_classes):
        super(SkeletonClassifier, self).__init__()
        dprob = 0.2
        self.network = nn.Sequential(nn.Linear(32*3, 128), nn.ReLU(), 
                                     nn.BatchNorm1d(128), 
                                     nn.Dropout(dprob),

                                     nn.Linear(128, 256), nn.ReLU(), 
                                     nn.BatchNorm1d(256), 
                                     nn.Dropout(dprob),

                                     nn.Linear(256, 512), nn.ReLU(), 
                                     nn.BatchNorm1d(512), 
                                     nn.Dropout(dprob),

                                     nn.Linear(512, 512), nn.ReLU(), 
                                     nn.BatchNorm1d(512), 
                                     nn.Dropout(dprob),

                                     nn.Linear(512, 256), nn.ReLU(), 
                                     nn.BatchNorm1d(256), 
                                     nn.Dropout(dprob),

                                     nn.Linear(256, 128), nn.ReLU(), 
                                    #  nn.BatchNorm1d(128), 
                                     ).to(torch.device('cuda'))
        self.logits = nn.Linear(128, num_classes).to(torch.device('cuda'))

    def forward(self, x):
        x = x.cuda()
        x = self.network(x)
        x = self.logits(x).cpu()
        return x
