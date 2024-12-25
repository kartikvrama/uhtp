import torch
from torch import nn

DEVICE = 'cpu'

class SkeletonClassifier(nn.Module):
    """Neural network classifier to predict activity labels from skeletal joint data."""
    def __init__(self, num_classes):
        super(SkeletonClassifier, self).__init__()

        dprob = 0.3

        self.network = nn.Sequential(nn.Linear(32*3, 512), nn.ReLU(), 
                                     nn.BatchNorm1d(512), 
                                     nn.Dropout(dprob),
                                     nn.Linear(512, 1024), nn.ReLU(), 
                                     nn.BatchNorm1d(1024), 
                                     nn.Dropout(dprob),
                                     nn.Linear(1024, 2048), nn.ReLU(), 
                                     nn.BatchNorm1d(2048), 
                                     nn.Dropout(dprob),
                                     nn.Linear(2048, 2048), nn.ReLU(), 
                                     nn.BatchNorm1d(2048), 
                                     nn.Dropout(dprob),
                                     nn.Linear(2048, 2048), nn.ReLU(), 
                                     nn.BatchNorm1d(2048), 
                                     nn.Dropout(dprob),
                                     nn.Linear(2048, 512), nn.ReLU(), 
                                     nn.BatchNorm1d(512), 
                                     ).to(torch.device(DEVICE))

        self.logits = nn.Linear(512, num_classes).to(torch.device(DEVICE))

    def forward(self, x):
        """Forward pass of the neural network."""
        x = x.to(torch.device(DEVICE))
        x = self.network(x)
        x = self.logits(x)
        return x
