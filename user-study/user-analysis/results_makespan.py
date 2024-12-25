import csv
from math import sqrt
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyexcel_ods import get_data

from scipy.stats import wilcoxon
from utils_results import ods_to_data

response_filter = ['n6', 'p1', 'p3', 'p4', 'p6', 'p7', 'p10', 'p12', 'p13', 
                   'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 
                   'p23','p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 
                   'p32', 'p33', 'p34']
    

def read_makespan_results(path, response_filter):
    odict = get_data(path)
    key = list(odict.keys())[0]

    rows = odict[key][1:]

    data = np.array([[r[1], r[2]] for r in rows if r[0] in response_filter])

    df = pd.concat([
                   pd.DataFrame(data=[[j[0], 'SUHTP'] for j in data], columns=['Task Completion Time', 'Mode']),
                   pd.DataFrame(data=[[k[1], 'SFIXED'] for k in data], columns=['Task Completion Time', 'Mode'])
                   ])

    return data, df

def main():
    opfile = './user_results/makespan_results.csv'

    # Column 0 is adaptive, column 1 is fixed
    data, df = read_makespan_results(opfile, response_filter=response_filter)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    dtrng = np.ptp(data, axis=0)

    print('Adaptive task execution time: Mean {:6.2f} and Std Dev {:6.3f}, Range {:6.2f}'.format(
            mean[0], std[0], dtrng[0]))
    print('Fixed task execution time: Mean {:6.2f} and Std Dev {:6.3f}, Range {:6.2f}'.format(
            mean[1], std[1], dtrng[1]))

    _, p = wilcoxon(data[:, 0], data[:, 1], zero_method="pratt", alternative="two-sided", 
                    mode="approx") # two tailed

    print('P value from Wicoxon Signed-Rank test two tailed for makespan (should be < 0.05): {:8.5f}'.format(p))

    # # fig = plt.figure(figsize=(16, 12))

    fig = plt.figure(figsize=(16, 9))
    colors = {'SUHTP': '#74add1', # uhtp
              'SFIXED': '#fe9929' # fixed
              }

    sns.set(font='Times New Roman')
    sns.set(font_scale=3)
    
    sns.set_style('whitegrid', {"grid.color":"0.1", "axes.edgecolor":"0.1"})

    # Box plot
    ax = sns.violinplot(data=df, y='Mode', x='Task Completion Time', orient='h', 
                        width=0.5, palette=colors)

    # # Scatter plot
    # ax = sns.swarmplot(data=df, y='Mode', x='Task Completion Time', color='.3')

    ax.set(xlim=[0, 700])
    ax.set(xlabel='Task Completion Time [secs] (lesser is better)')

    plt.yticks(rotation=0, multialignment='center')

    # plt.show()

    fig.savefig('./plots/makespan_comparison.png', bbox_inches='tight')



if __name__ == '__main__':
    main()