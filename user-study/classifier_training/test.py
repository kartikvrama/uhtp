import os
import yaml
import glob
import argparse

import cv2
import torch
import numpy as np

from utils import *
from model import DEVICE, SkeletonClassifier

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def my_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax = e_x / np.sum(e_x, axis=1, keepdims=True)
    return softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_config')

    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    with open("dataset/metadata.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    # Gather class labels
    label_dict = data_config['label_dict']
    class_names = list(label_dict.keys())

    # Fetch test files
    file_list = glob.glob(os.path.join(data_config['test_data_dir'], '{}'.format(data_config['test_data_tag'])))
    print(file_list)

    # Test config and pelvis mean
    test_config = model_config['Test']
    pelvis_mean = np.load(test_config['pelvis_mean'])

    # Test data (tensor) and labels
    test_data, test_labels, _, _, _, _ = load_data(file_list, num_classes=data_config['num_classes'], 
                                                   mean=pelvis_mean, shuffle=False, test_len=0)
    test_data = torch.Tensor(test_data).to(torch.device(DEVICE))

    # Load model to device in forward mode (eval)
    model = SkeletonClassifier(num_classes=len(class_names))
    model.eval()

    # Restore checkpoint
    model.load_state_dict(torch.load(test_config['test_ckpt'], map_location=torch.device(DEVICE)))    

    # Forward pass
    test_logits = model.forward(test_data).detach().cpu().numpy()

    # Predictions using argmax
    new_preds = np.argmax(test_logits, axis=-1).astype(np.int32)

    # Plot confusion matrix
    generate_and_plot_confusion_matrix(new_preds, test_labels, class_names, True)


if __name__ == '__main__':
    np.random.seed(12345)
    torch.random.manual_seed(12345)
    main()
