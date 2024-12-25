import os
import yaml
import numpy as np

import torch
from action_classifier.model import SkeletonClassifier


class ActionPred():
    def __init__(self, model_path):
        with open(os.path.join(model_path, 'model_config.yaml')) as f:
            config = yaml.load(f)
        self.pred2class = config['class_names']
        input_dim = config['input_dim']
        num_joints = input_dim//3

        self.model = SkeletonClassifier(num_classes=len(self.pred2class), dim=input_dim).cuda()
        self.model.load_state_dict(torch.load(os.path.join(model_path, config['eval_model'])))
        self.model.eval()

        self.pelvis_mean = np.load(os.path.join(model_path, 'pelvis_mean.npy'))
        self.pelvis_mean = np.tile(self.pelvis_mean, num_joints)

    def predict_action(self, data):

        # # Ignoring nose and ears, remove if using original model
        # data = data[:-5]

        data = data.ravel() - self.pelvis_mean
        data = torch.Tensor(data).cuda()
        logits = self.model.forward(data.view(1, -1)).detach().cpu().numpy()
        pred = int(np.argmax(logits.ravel()))
        return self.pred2class[pred]
