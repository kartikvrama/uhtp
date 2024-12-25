import torch
from torch import nn

class SkeletonClassifier(nn.Module):
    """
    the neural network classifier for skeletal joint data
    """
    def __init__(self, num_classes, dim, **kwargs):
        super(SkeletonClassifier, self).__init__()
        dprob = 0.3
        self.network = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), 
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
                                     ).to(torch.device('cuda'))
        self.logits = nn.Linear(512, num_classes).to(torch.device('cuda'))

    def forward(self, x):
        # x = x.view(len(x), 32*3)
        assert(x.shape == (len(x), 32*3))
        x = x.cuda()
        x = self.network(x)
        x = self.logits(x).cpu()
        return x
