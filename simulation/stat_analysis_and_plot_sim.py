import argparse
import pickle as pkl
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

from scipy.stats import mannwhitneyu as mwu

parser=argparse.ArgumentParser()
parser.add_argument("--task", help="Specify task as ikea or drill")
ARGS=parser.parse_args()


def sns_change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def ikea_chair():

    # Load data
    with open('./results/ikea_chair_results.pickle', 'rb') as handle:
        makespan_dict = pkl.load(handle)

    for rbid, robot_behav in enumerate(['uhtp', 'random', 'greedy', 'fixed']):

            result = makespan_dict[rbid]

            print(robot_behav)
            print('\tMin: ' + str(min(result)))
            print('\tMax: ' + str(max(result)))
            print('\tAverage: ' + str(sum(result)/len(result)))
            print('\tStd: ' + str(np.std(np.array(result))))
            print('')

    rbdict = {'uhtp':0, 'fixed':3, 'greedy':2, 'random':1}

    # Compare uhtp with fixed
    _, p = mwu(makespan_dict[rbdict['uhtp']], makespan_dict[rbdict['fixed']]) # other params set to default

    print('UHTP vs Fixed, p value {}'.format(p))

    # Compare uhtp with greedy
    _, p = mwu(makespan_dict[rbdict['uhtp']], makespan_dict[rbdict['greedy']]) # other params set to default

    print('UHTP vs Greedy, p value {}'.format(p))

    # Compare uhtp with random
    _, p = mwu(makespan_dict[rbdict['uhtp']], makespan_dict[rbdict['random']]) # other params set to default

    print('UHTP vs Random, p value {}'.format(p))

    # Compare fixed with greedy
    _, p = mwu(makespan_dict[rbdict['fixed']], makespan_dict[rbdict['greedy']]) # other params set to default

    print('Fixed vs Greedy, p value {}'.format(p))

    # Compare fixed with random
    _, p = mwu(makespan_dict[rbdict['fixed']], makespan_dict[rbdict['random']]) # other params set to default

    print('Fixed vs Random, p value {}'.format(p))

    # Plot results

    # fixed-#fc8d59, greedy-#b2df8a
    colors = {
        'uhtp': '#74add1', 'fixed': '#fe9929', 'random': '#af8dc3', 'greedy': '#78c679'
    } 

    N = len(makespan_dict[0])

    # Append items to pd dataframe
    df = pd.DataFrame(data={})

    for key, value in rbdict.items():

        d1 = pd.DataFrame([[makespan_dict[value][i], key] for i in range(N)],
                                columns = ['Data', 'Robot Behavior'])
        df = pd.concat([df, d1])

    fig = plt.figure(figsize=(12, 12))

    sns.set(font='Times New Roman')
    sns.set(font_scale=2)

    sns.set_style('whitegrid', {"grid.color":"0.6", "axes.edgecolor":"0.6"})

    # Bar plot
    ax = sns.boxplot(data=df, x='Robot Behavior', y='Data', #orient='h',
                        palette=colors, width=0.5)

    # sns_change_width(ax, 0.5)

    ax.set(ylim=[0, 120])
    ax.set(yticks=range(0, 125, 10))
    ax.set(xlabel=None, ylabel=None)
    ax.set(xticks=[])

    # plt.show()

    fig.tight_layout()
    fig.savefig('./plots/chair_assembly_plot', bbox_inches='tight')

    return

def drill_assm():
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = 20

    pfail_range = np.arange(0, 1.01, 0.1)

    robot_behaviors = ['uhtp', 'random', 'greedy', 'fixed']

    # fixed-(252, 141, 89), greedy-(178, 223, 138)
    colors_rb = {
            'uhtp': (116, 173, 209), 
            'fixed': (217,95,14), 
            'random': (175, 141, 195), 
            'greedy': (49,163,84)
        } 

    colors_rb = {key: [colors_rb[key][i]/255.0 for i in range(3)] for key in colors_rb.keys()}

    fig = plt.figure(figsize=(12, 9))

    ax = plt.axes()
    ax.grid(False)

    ax.set(xlim=[0, 1])
    ax.set(ylim=[100, 200])
    # ax.set(font='Times New Roman')

    ax.set(xticks=pfail_range)
    ax.set(ylabel='Task Completion Time [secs]', xlabel='Probability of Quality Control Failure')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for rbid, robot_behav in enumerate(robot_behaviors):

        # Load data
        with open('./results/drill_assm_algo_{}_results.pickle'.format(robot_behav), 'rb') as handle:
            makespan_rb_dict = pkl.load(handle)

        num_trials = 1000

        makespan_rb_df_mean = np.array([])
        makespan_rb_df_std = np.array([])

        for pfail in pfail_range:

            makespan_rb_pfail_array = makespan_rb_dict[str(pfail)]

            makespan_rb_df_mean = np.concatenate([makespan_rb_df_mean, [np.mean(makespan_rb_pfail_array)]])
            makespan_rb_df_std = np.concatenate([makespan_rb_df_std, [np.std(makespan_rb_pfail_array)]])

        ax.plot(pfail_range, makespan_rb_df_mean, '-', color=colors_rb[robot_behav], label=robot_behav)
        ax.fill_between(pfail_range, makespan_rb_df_mean - makespan_rb_df_std, makespan_rb_df_mean + makespan_rb_df_std,
                            color=colors_rb[robot_behav]+[0.2])

    ax.legend(edgecolor='1.0')

    fig.tight_layout()
    fig.savefig('./plots/drill_assembly_plot', bbox_inches='tight')

    return

if __name__ == '__main__':

    if ARGS.task == "ikea":
        ikea_chair()
    elif ARGS.task == "drill":
        drill_assm()
    else:
        raise ValueError(f"Expected task to be either ikea or drill, instead received {ARGS.task}")
