import numpy as np
import matplotlib.pyplot as plt

from utils import load_data

def return_lines(vector, skeleton):
    xline = []
    yline = []
    zline = []
    for i in range(1, len(vector)):
        xline.append(skeleton[i, 0] - skeleton[i-1, 0])
        yline.append(skeleton[i, 1] - skeleton[i-1, 1])
        zline.append(skeleton[i, 2] - skeleton[i-1, 2])
    return xline, yline, zline

def return_points(vector, skeleton):
    xline = []
    yline = []
    zline = []
    for i in vector:
        xline.append(skeleton[i, 0])# - skeleton[i-1, 0])
        yline.append(skeleton[i, 1])# - skeleton[i-1, 1])
        zline.append(skeleton[i, 2])# - skeleton[i-1, 2])
    return xline, yline, zline


def plot_skeleton(skeleton, title):
    spine_vec = [0, 1, 2, 3, 26]
    left_hand_vec = [2, 4, 5, 6, 7, 8, 9, 10] # 10 is left thumb
    right_hand_vec = [2, 11, 12, 13, 14, 15, 16, 17] # 17 is right thumb
    left_leg_vec = [0, 18, 19, 20, 21]
    right_leg_vec = [0, 22, 23, 24, 25]
    all_vecs = [spine_vec, left_hand_vec, right_hand_vec, left_leg_vec, right_leg_vec]
    colors = ['g', 'black', 'blue', 'black', 'blue']

    xlines = []
    ylines = []
    zlines = []

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for n, vec in enumerate(all_vecs):
        x, y, z = return_points(vec, skeleton)
        ax.plot(x, y, z, color=colors[n])
        xlines += x
        ylines += y
        zlines += z

    ax.plot(skeleton[0, 0], skeleton[0, 1], skeleton[0, 2], 'ro') # pelvis
    ax.plot(skeleton[26, 0], skeleton[26, 1], skeleton[26, 2], 'r+') # neck

    ax.set_xlabel("X+")
    ax.set_ylabel("Y+")
    ax.set_zlabel("Z+")
    ax.set_title(title)


def reflect(skeleton):
    spine_vec = [0, 1, 2, 3, 26]
    left_hand_vec = [2, 4, 5, 6, 7, 8, 9, 10] 
    right_hand_vec = [2, 11, 12, 13, 14, 15, 16, 17] 
    left_leg_vec = [0, 18, 19, 20, 21]
    right_leg_vec = [0, 22, 23, 24, 25]

    pelvis_x = skeleton[0, 0]
    skeleton_x = skeleton[:, 0] + 2*(pelvis_x - skeleton[:, 0])
    skeleton[:, 0] = skeleton_x

    new_skeleton = np.zeros((32, 3))
    new_skeleton[spine_vec, :] = skeleton[spine_vec, :]

    new_skeleton[left_hand_vec, :] = skeleton[right_hand_vec, :]
    new_skeleton[right_hand_vec, :] = skeleton[left_hand_vec, :]

    new_skeleton[left_leg_vec, :] = skeleton[right_leg_vec, :]
    new_skeleton[right_leg_vec, :] = skeleton[left_leg_vec, :]

    return new_skeleton

if __name__ == '__main__':
    filename = 'data_withrobot_extended/val/recording1_sonia-annotated.csv'
    data = load_data([filename], 6, mean=np.zeros(3), test_len=0)[0]
    
    skeleton = data[955].reshape(-1, 3)
    new_skeleton = reflect(skeleton)
    plot_skeleton(skeleton, 'old')
    plot_skeleton(new_skeleton, 'new')

    plt.show()
