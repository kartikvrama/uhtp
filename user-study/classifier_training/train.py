import os
import yaml
import glob
import argparse

import torch
import numpy as np
from torch import nn

from utils import load_data, shuffle_data, generate_confusion_matrix, plot_confusion_matrix
from model import SkeletonClassifier
from focal_loss import FocalLoss, reweight

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: {}'.format(DEVICE))

def accuracy(logits, labels):
    batch_size = len(labels)
    _, pred = torch.max(logits, dim=-1)

    correct = pred.eq(labels).sum() * 1.0
    acc = (correct/batch_size).detach().cpu().numpy()
    return acc


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_config')

    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    with open("dataset/metadata.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['class_names']

    file_list = glob.glob(os.path.join(data_config['train_data_dir'], '{}'.format(data_config['train_data_tag'])))

    train_data, train_labels, _, _, \
    cls_num_list, pelvis_mean = load_data(file_list, num_classes=data_config['num_classes'],
                                                       shuffle=True, test_len=0)

    test_file_list = glob.glob(os.path.join(data_config['val_data_dir'], '{}'.format(data_config['val_data_tag'])))
    test_data, test_labels, _, _, _, _ = load_data(test_file_list, num_classes=data_config['num_classes'], 
                                                   mean=pelvis_mean, shuffle=False, test_len=0)

    model = SkeletonClassifier(num_classes=data_config['num_classes'])
    model = model.to(DEVICE)

    per_cls_weights = reweight(cls_num_list, beta=model_config['beta'])
    if model_config['loss_function'] == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1.)
    elif model_config['loss_function'] == 'Cross-entropy-weighted':
        per_cls_weights = torch.Tensor(per_cls_weights)
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
    elif model_config['loss_function'] == 'Cross-entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=float(model_config['learning_rate']),
                                 weight_decay=float(model_config['weight_decay'])
                                 )

    test_data = torch.Tensor(test_data)
    train_len = len(train_data)

    train_loss_history = []
    val_acc_history = []

    if not os.path.isdir('./{}'.format(model_config['logname'])):
        os.mkdir('./{}'.format(model_config['logname']))

    print(cls_num_list)

    best_confusion_matrix = None
    for epoch in range(model_config['num_epochs']):
        train_acc = 0
        num_batches = train_len//model_config['batch_size']

        for idx in range(num_batches):
            batch_data = torch.Tensor(train_data[idx*model_config['batch_size']:(idx + 1)*model_config['batch_size'], :])
            batch_labels = torch.LongTensor(train_labels[idx*model_config['batch_size']:(idx + 1)*model_config['batch_size']])
            
            optimizer.zero_grad()
            batch_logits = model.forward(batch_data)
            loss = criterion(input=batch_logits, target=batch_labels)
            train_loss_history.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

            batch_train_acc = accuracy(batch_logits, batch_labels)
            train_acc += batch_train_acc

        train_acc /= num_batches

        model.eval()
        with torch.no_grad():
            test_logits = model.forward(test_data).detach().cpu().numpy()
        model.train()

        confusion_matrix = generate_confusion_matrix(test_logits, test_labels, data_config['num_classes'], True)

        # Remove last label (grab_parts) from confusion matrix.
        working_confusion_matrix = confusion_matrix[:-1, :-1]
        working_confusion_matrix /= np.sum(working_confusion_matrix, axis=1, keepdims=True)

        test_acc = np.mean(np.diag(working_confusion_matrix))
        full_test_acc = np.mean(np.diag(confusion_matrix))

        val_acc_history.append(test_acc)
        print('Epoch: {:4d} | Train Acc: {:6.3f} | Working Val Acc: {:6.3f} | Full Val Acc: {:6.3f}'.format(epoch+1, train_acc, test_acc, full_test_acc))
        train_data, train_labels = shuffle_data(train_data, train_labels)

        if epoch > 10:
            if test_acc > max(val_acc_history[:-1]):
                print('Best Acc so far: {:6.3f}'.format(test_acc))
                torch.save(model.state_dict(), './{}/model_front_bestacc.ckpt'.format(model_config['logname'], epoch+1))
                best_confusion_matrix = working_confusion_matrix
            elif (epoch+1)%100 == 0:
                torch.save(model.state_dict(), './{}/model_front_{}.ckpt'.format(model_config['logname'], epoch+1))

    print("Class names: ", class_names)
    if best_confusion_matrix is not None:
        print("Best confusion matrix\n", best_confusion_matrix)
        plot_confusion_matrix(best_confusion_matrix, class_names[:-1])

    np.save('./{}/pelvis_mean.npy'.format(model_config['logname']), pelvis_mean)
    np.save('./{}/loss_history'.format(model_config['logname']), train_loss_history)
    np.save('./{}/val_acc_history'.format(model_config['logname']), val_acc_history)

    torch.save(model.state_dict(), './{}/model_front_final.ckpt'.format(model_config['logname']))
    with open('./{}/trained_config.yaml'.format(model_config['logname']), 'w') as dumpfile:
        yaml.dump(model_config, dumpfile)

if __name__ == '__main__':
    np.random.seed(12345)
    torch.random.manual_seed(12345)
    train()
