import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import f1_score
from torch._C import dtype


def generate_and_plot_confusion_matrix(preds, labels, class_names, normalize=True):
    """ Create confusion matrix and save the figure """
    num_classes = len(class_names)

    confusion_matrix = np.zeros((num_classes, num_classes))

    for target, prediction in zip(labels, preds):
        confusion_matrix[target, prediction] += 1

    print(confusion_matrix.astype(np.int32))

    if normalize:
        confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)

    print('Average accuracy: {:6.3f}'.format(100*np.mean(np.diag(confusion_matrix))))
    print('F1 score: {:6.4f}'.format(f1_score(y_true=labels, y_pred=preds, average="macro")))
    print(f1_score(y_true=labels, y_pred=preds, average=None))
    plot_confusion_matrix(confusion_matrix, class_names)


def my_softmax(x):
    """ Compute softmax values for each sets of scores in x. """

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_to_pretty_string(softmax):
    """
    given the softmax readings, convert it into a pretty string to be put on the video
    :param softmax:
    :return:
    """

    softmax_ints = (softmax*50).astype(int)
    str_bars = []
    for i, v in enumerate(softmax_ints):
        str_bars.append('*'*v + '*')
    return str_bars


def write_hud(frame, action, softmax, class_names):
    c = (0, 255, 0)
    action_str = class_names[action]
    softmax_strs = softmax_to_pretty_string(softmax)
    frame = cv2.resize(frame, (640*2, 480*2))
    frame2 = frame.copy()
    frame2 = cv2.rectangle(frame2, (0, 0), (800, 125), (0,0,0), -1)
    frame2 = cv2.rectangle(frame2, (445*2,0),(640*2, 18*25+55),(0,0,0), -1)
    frame = cv2.addWeighted(frame, .5, frame2, .5, 0)
    frame = cv2.putText(frame, 'current action- '+str(action)+' '+action_str, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2)

    frame = cv2.putText(frame, 'softmax info:', (450*2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for i in range(len(softmax)):
        frame = cv2.putText(frame, class_names[i], (450*2, 55+i*25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        frame = cv2.putText(frame, str(softmax_strs[i]), (525*2, 55+i*25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    return frame


def generate_confusion_matrix(logits, labels, num_classes, normalize=True):
    preds = np.argmax(logits, axis=-1)

    confusion_matrix = np.zeros((num_classes, num_classes))
    for target, prediction in zip(labels, preds):
        confusion_matrix[target, prediction] += 1
    if normalize:
        confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, class_labels):
    fig, ax = plt.subplots()
    fig.set_figheight(11)
    fig.set_figwidth(11)

    # print(rcParams.keys())

    # rcParams['font.family'] = ['serif']
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 16

    num_classes = len(class_labels)

    clean_class_labels = ['Attach Shell', 'Screw', 'Attach Battery',
                            'Place Drill', 'Null', 'Grab Parts']

    ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(clean_class_labels)
    ax.set_yticklabels(clean_class_labels)

    # ax.set_xlabel("Predicted Label")
    # ax.set_ylabel("Ground-Truth label")
    # ax.set_title("Confusion Matrix")

    plt.setp(
        ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            _ = ax.text(
                j,
                i,
                "{:.2f}".format(round(confusion_matrix[i, j], 2)),
                ha="center",
                va="center",
                color="black",
            )
    # plt.show()
    plt.savefig('./confusion_matrix.png')


def plot_npfile(file, show=True):
    data = np.load(file, allow_pickle=True)
    plt.figure()
    plt.plot(data)
    if show:
        plt.show()


def shuffle_data(data, labels):
    combined_array = np.hstack([data, labels[:, np.newaxis]])
    np.random.shuffle(combined_array)
    data = combined_array[:, :data.shape[1]]
    labels = combined_array[:, data.shape[1]:].ravel()
    labels = labels.astype(np.int32)
    return data, labels


def load_data_test_invalids(file_list, num_classes, mean=None):
    labels = [] 
    invalids = []
    data = []
    for filename in file_list:
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                labels.append(int(float(row[0])))
                invalids.append([int(j) for j in row[1].split(',')])
                data.append([float(j) for j in row[2:]])

    data = np.array(data)
    labels = np.array(labels, dtype=np.int32).ravel()
    if mean is None:
        pelvis_mean = np.mean(data[:, :3], axis=0)
    else:
        pelvis_mean = np.array(mean)
    data = data - np.tile(pelvis_mean, 32)

    cls_num_list = []
    for i in range(num_classes):
        cls_idxs = np.where(labels==i)[0]
        cls_num_list.append(len(cls_idxs))

    return data, labels, invalids, cls_num_list, pelvis_mean


def load_data(file_list, num_classes, mean=None, shuffle=True, test_len=140):
    labels = []
    data = []
    for filename in file_list:
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                labels.append(int(row[0]))
                # data.append([float(j) for j in row[1:-1]])
                data.append([float(j) for j in row[1:-1]])

    data = np.array(data, dtype=np.float64)
    labels = np.array(labels, dtype=np.int32).ravel()
    if mean is None:
        pelvis_mean = np.mean(data[:, :3], axis=0)
    else:
        pelvis_mean = np.array(mean)
    data = data - np.tile(pelvis_mean, 32)#np.tile(data[:, :3], (1, 32))#np.tile(pelvis_mean, 32)

    if shuffle:
        combined_array = np.hstack([data, labels[:, np.newaxis]])
        np.random.shuffle(combined_array)
        data = combined_array[:, :data.shape[1]]
        labels = combined_array[:, data.shape[1]:].ravel()
        labels = labels.astype(np.int32)

    test_labels = []
    test_data = np.zeros([test_len, data.shape[1]])

    cls_num_list = []
    len_per_class = test_len//num_classes
    for i in range(num_classes):
        cls_idxs = np.where(labels==i)[0]
        cls_num_list.append(len(cls_idxs))

        if test_len:
            test_data[i*len_per_class:(i+1)*len_per_class, :] = data[cls_idxs[:len_per_class], :]
            data = np.delete(data, cls_idxs[:len_per_class], 0)

            test_labels.append(labels[cls_idxs[:len_per_class]])
            labels = np.delete(labels, cls_idxs[:len_per_class])
    test_labels = np.array(test_labels, dtype=np.int32).ravel()

    return data, labels, test_data, test_labels, cls_num_list, pelvis_mean


def load_data_trunc(file_list, num_classes, mean=None, shuffle=True, test_len=140):
    labels = []
    data = []

    # Remove Nose and Ears
    remove_joints = np.arange(27*3, 32*3)

    # # Removing joints
    # remove_joints = np.concatenate([np.arange(18*3, 26*3), 
    #                                 np.arange(27*3, 32*3)]).astype(np.int32)

    for filename in file_list:
        with open(filename, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                labels.append(int(row[0]))
                skeleton = [float(j) for j in row[1:-1]]
                skeleton = np.delete(skeleton, remove_joints)
                data.append(skeleton)

    data = np.array(data, dtype=np.float64)
    num_joints = data.shape[-1]//3
    labels = np.array(labels, dtype=np.int32).ravel()
    if mean is None:
        pelvis_mean = np.mean(data[:, :3], axis=0)
    else:
        pelvis_mean = np.array(mean)
    data = data - np.tile(pelvis_mean, num_joints)#np.tile(data[:, :3], (1, 32))#np.tile(pelvis_mean, 32)

    if shuffle:
        combined_array = np.hstack([data, labels[:, np.newaxis]])
        np.random.shuffle(combined_array)
        data = combined_array[:, :data.shape[1]]
        labels = combined_array[:, data.shape[1]:].ravel()
        labels = labels.astype(np.int32)

    test_labels = []
    test_data = np.zeros([test_len, data.shape[1]])

    cls_num_list = []
    len_per_class = test_len//num_classes
    for i in range(num_classes):
        cls_idxs = np.where(labels==i)[0]
        cls_num_list.append(len(cls_idxs))

        if test_len:
            test_data[i*len_per_class:(i+1)*len_per_class, :] = data[cls_idxs[:len_per_class], :]
            data = np.delete(data, cls_idxs[:len_per_class], 0)

            test_labels.append(labels[cls_idxs[:len_per_class]])
            labels = np.delete(labels, cls_idxs[:len_per_class])
    test_labels = np.array(test_labels, dtype=np.int32).ravel()

    return data, labels, test_data, test_labels, cls_num_list, pelvis_mean


if __name__ == '__main__':
    import glob
    file_list = glob.glob('extracted_data/*-annotated.csv')
    data, labels, test_data, test_labels, cls_num_list = load_data(file_list, 7)
    print(cls_num_list)
