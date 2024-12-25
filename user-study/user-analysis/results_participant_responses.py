import argparse

import numpy as np
from math import sqrt
from copy import deepcopy
from pandas.core.frame import DataFrame

from scipy.stats import mode as mode_fn, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from utils_results import read_bernoulli, ods_to_data, read_4pt_likert, argsort_list, sns_change_width

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Only consider responses from the following participant IDs.
response_filter = [5, 6, 8, 9, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24, 
                   25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

bool_to_score = {'Yes': 1, 'No': 2}
likert_to_score = {'Always': 4, 'Most of the Time': 3, 'Sometimes': 2, 'Never': 1}
reverselikert_to_score = {'Always': 1, 'Most of the Time': 2, 'Sometimes': 3, 'Never': 4}

binary_columns = [1, 6]
likert_columns = [2, 3, 4, 5]
tlx_columns = [7, 8, 9, 10, 11, 12]

run_comp_columns = [1, 2, 3]

SAVEFOL = 'plots'

parser=argparse.ArgumentParser()
parser.add_argument("--likert", type=bool, default=False, help="Whether to analyze likert respones")
parser.add_argument("--binary", type=bool, default=False, help="Whether to analyze binary responses")
parser.add_argument("--tlx", type=bool, default=False, help="Whether to analyze NASA TLX responses")
parser.add_argument("--post_expt", type=bool, default=False, help="Whether to analyze post experiment responses")
ARGS = parser.parse_args()


def demographics(columns, data):

    #Age find median and range

    age = data[1]

    print('Age Min {}, Max {}, Range {}, Median {}'.format(min(age), max(age), (max(age) - min(age)), np.median(age)))

    print('\n\n')
    for dgidx in range(2, len(columns)):

        print('Question: {}'.format(columns[dgidx]))

        items, counts = np.unique(data[dgidx], return_counts=True)

        for itm, frq in zip(items, counts):
            print('{}: {}'.format(itm, frq))

        print('\n')


def survey_questions(columns, responses_adap, responses_fixed):

    print('\n\n')

    # Evaluate Likert scale answers
    print('Likert items')

    N = len(responses_adap[0])
    df = pd.DataFrame(data={})

    question_labels = ['RAcc*', 'RFR', 'RID', 'RIA']
    file_names = ['q1', 'q2', 'q3', 'q4']

    comparisons = ['two-sided', 'two-sided', 'two-sided', 'two-sided'] # two-tailed
    answer_count_adap = {}
    answer_count_fixed = {}

    for j, cidx in enumerate(likert_columns):
        answer_count_adap[question_labels[j]] = {i:0 for i in range(1, 5)}
        answer_count_fixed[question_labels[j]] = {i:0 for i in range(1, 5)}

        if j == 0: # RAcc*
            scores_adap, median_adap, binc_adap = read_4pt_likert(responses_adap[cidx], reverselikert_to_score)
            scores_fixed, median_fixed, binc_fixed = read_4pt_likert(responses_fixed[cidx], reverselikert_to_score)

        else:
            scores_adap, median_adap, binc_adap = read_4pt_likert(responses_adap[cidx], likert_to_score)
            scores_fixed, median_fixed, binc_fixed = read_4pt_likert(responses_fixed[cidx], likert_to_score)

        print('Original median for Adaptive: {}, Fixed: {}'.format(median_adap, median_fixed))

        # Wilcoxon Signed-Rank non-parametric test for paired samples- testing if adap response is DIFFERENT than fixed response
        _, p = wilcoxon(scores_adap, scores_fixed, zero_method="pratt", alternative=comparisons[j], 
                        mode="approx", correction=True)

        print('P value from Wicoxon Signed-Rank test two tailed for {} score (should be < 0.05): {:8.5f}'.format(comparisons[j], p))
        print('')

        # Collecting bin counts in 
        for key in binc_adap:
            answer_count_adap[question_labels[j]][key] = binc_adap[key]

        for key in binc_fixed:
            answer_count_fixed[question_labels[j]][key] = binc_fixed[key]

    responses = ['Never', 'Sometimes', 'Most of the Time', 'Always']

    width = 0.075
    x = [0.2, 0.2+width] #['Adaptive\nHTN', 'Fixed\nSequence']

    for idx, q in enumerate(question_labels):

        num_participants = N

        # Centering adaptive and fixed behavior data b/w positive and negative responses

        data_adap = [answer_count_adap[q][k] for k in range(1, 5)]
        data_adap.insert(0, num_participants - data_adap[0] - data_adap[1])
        data_adap.insert(5, num_participants - data_adap[0])

        data_fixed = [answer_count_fixed[q][k] for k in range(1, 5)]
        data_fixed.insert(0, num_participants - data_fixed[0] - data_fixed[1])
        data_fixed.insert(5, num_participants - data_fixed[0])

        ## Plot Adaptive v/s Fixed frequency data
        y_data = np.vstack([data_adap, data_fixed]).T

        # Bottom buffer
        y_prev = y_data[0, :]
        plt.bar(x, y_prev, label='1', alpha=0.0, width=width)

        color_adap = [(189,201,225),
                      (116,169,207),
                      (43,140,190),
                      (4,90,141)]

        color_fixed = [(254,217,142),
                       (254,153,41),
                       (217,95,14),
                       (153,52,4)]

        # colors = [(202/255.0,0/255.0,32/255.0), 
        #           (244/255.0,165/255.0,130/255.0), 
        #           (146/255.0,197/255.0,222/255.0), 
        #           (5/255.0,113/255.0,176/255.0)]

        alphas = [1, 1, 1, 1]

        fig=plt.figure(figsize=(9, 16))

        sns.set_style('whitegrid', {"grid.color":"0", "axes.edgecolor":"0"})

        axes = plt.gca()
        axes.xaxis.grid(False)

        # Stacking bar plots on top of previous bar
        for i in range(1, len(data_adap)-1):

            y_curr = y_data[i, :]

            axes.bar(x[0], y_curr[0], bottom=y_prev[0], label=responses[i-1], 
                        color=[c/255.0 for c in color_adap[i-1]], 
                        alpha=alphas[i-1], width=width)

            axes.bar(x[1], y_curr[1], bottom=y_prev[1], label=responses[i-1], 
                        color=[c/255.0 for c in color_fixed[i-1]], 
                        alpha=alphas[i-1], width=width)

            y_prev += np.array(y_curr)

            # Top buffer
            y_curr = y_data[-1, :]
            axes.bar(x, y_curr, bottom=y_prev, label='2', alpha=0.0, width=width)

            # Axis formatting
            axes.set(ylim=[0, 2*num_participants])

            axes.set_xticks(x)
            axes.set_xticklabels(['Adaptive\nHTN', 'Fixed\nSequence'])

            axes.set_xlim(0, 1)

        fig.savefig('./{}/likert_{}'.format(SAVEFOL, file_names[idx]), bbox_inches='tight')

        print('')


def tlx_score(columns, responses_adap, responses_fixed):

    print('\n\n')

    # Calculating tlx score
    print('Final TLX score')

    tlx_score_adap = np.array([responses_adap[i] for i in tlx_columns])
    tlx_score_adap = np.sum(tlx_score_adap, axis=0)

    tlx_score_fixed = np.array([responses_fixed[i] for i in tlx_columns])
    tlx_score_fixed = np.sum(tlx_score_fixed, axis=0)

    N = len(tlx_score_adap)

    median_adap = np.median(tlx_score_adap)
    median_fixed = np.median(tlx_score_fixed)

    print('Adaptive--> Mean: {:6.3f}, Stdev: {:6.3f}'.format(np.mean(tlx_score_adap), np.std(tlx_score_adap)))
    print('Fixed--> Mean: {:6.3f}, Stdev: {:6.3f}'.format(np.mean(tlx_score_fixed), np.std(tlx_score_fixed)))
    print('Median for Fixed: {}, Adaptive: {}'.format(median_fixed, median_adap))

    _, p = wilcoxon(tlx_score_adap, tlx_score_fixed, zero_method="pratt", alternative="less", 
                    mode="approx", correction=True)
    print('P value from Wilcoxon Signed-rank test (should be < 0.05): {:8.5f}'.format(p))

    print('\n')

    d_adap = pd.DataFrame([[tlx_score_adap[i], 'SUHTP', ''] for i in range(N)],
                            columns = ['Data', 'Robot Behavior', 'Question'])

    d_fixed = pd.DataFrame([[tlx_score_fixed[i], 'SFIXED', ''] for i in range(N)],
                            columns = ['Data', 'Robot Behavior', 'Question'])

    df = pd.concat([d_adap, d_fixed])

    fig = plt.figure(figsize=(16, 9))
    colors = {'SUHTP': '#74add1', # uhtp
              'SFIXED': '#fe9929' # fixed
              }

    sns.set(font='Times New Roman')
    sns.set(font_scale=3)
    
    sns.set_style('whitegrid', {"grid.color":"0.1", "axes.edgecolor":"0.1"})

    # Bar plot
    ax = sns.violinplot(data=df, y='Question', x='Data', hue='Robot Behavior', 
                        orient='h', width=0.5, palette=colors) 
                        #, width=0.5, dodge=0.1)

    # sns_change_width(ax, 0.1)

    ax.set(xlim=[0, 24])
    ax.set_xticks(range(0, 25, 2))
    ax.set(ylabel=None, xlabel="NASA TLX Score (lesser is better)")
    ax.get_legend().remove()

    fig.tight_layout()
    fig.savefig('./{}/tlx_scores'.format(SAVEFOL), bbox_inches='tight')


def binary_questions(columns, responses_adap, responses_fixed):
    """ Evaluate binary response answers
    """

    print('\n\n')

    print('Binary questions')

    N = len(responses_adap[0])
    df = pd.DataFrame(data={})

    question_labels = ['Did the robot always \n track drill color?', 'Does the robot adapt \n to user actions?']

    frequencies = []
    percentages = []

    for j, cidx in enumerate(binary_columns):

        print('Question: {}'.format(columns[cidx]))

        scores_adap, mode_adap = read_bernoulli(responses_adap[cidx], bool_to_score)
        scores_fixed, mode_fixed = read_bernoulli(responses_fixed[cidx], bool_to_score)

        # Populate frequencies for contingency table
        adap_yes = len([s for s in scores_adap if s == bool_to_score['Yes']])
        adap_no = len([s for s in scores_adap if s == bool_to_score['No']])

        fixed_yes = len([s for s in scores_fixed if s == bool_to_score['Yes']])
        fixed_no = len([s for s in scores_fixed if s == bool_to_score['No']])

        frequencies.append([adap_yes, fixed_yes])

        # Percentage of yes response 
        percent_adap =  100*adap_yes/(adap_yes + adap_no)
        percent_fixed = 100*fixed_yes/(fixed_yes + fixed_no)

        print('Percentage of yes: Fixed {:3.1f} and Adaptive {:3.1f}'.format(percent_fixed, percent_adap))
        print('Odds ratio Adap vs Fixed: {:5.2f}'.format((adap_yes/adap_no)/(fixed_yes/fixed_no)))

        # Generate pd dataframe
        d_adap = pd.DataFrame([[responses_adap[cidx][i], 'uhtp', question_labels[j]] 
                                for i in range(N) if responses_adap[cidx][i]=='Yes'],
                                columns = ['Data', 'Robot Behavior', 'Question'])
        df = df.append(d_adap, ignore_index=True)

        d_fixed = pd.DataFrame([[responses_fixed[cidx][i], 'fixed', question_labels[j]] 
                                for i in range(N) if responses_fixed[cidx][i]=='Yes'],
                                columns = ['Data', 'Robot Behavior', 'Question'])
        df = df.append(d_fixed, ignore_index=True)

        print('\n')

    fig = plt.figure(figsize=(12, 12))

    sns.set(font='Times New Roman')
    sns.set(font_scale=2)

    sns.set_style('whitegrid', {"grid.color":"0.6", "axes.edgecolor":"0.6"})

    colors = {
        'uhtp': '#74add1', 'fixed': '#fe9929'
    }

    # Plot histogram comparing adaptive to fixed behavior
    ax = sns.histplot(data=df, x='Question', hue='Robot Behavior', stat='count', multiple='dodge', 
                        shrink=.8, palette=colors, alpha=1.0, legend=False)

    for i, f in enumerate(frequencies):
        # ax.text(i-0.2, f[0]-0.8, round(100*f[0]/N, 1), color='black', ha="center", fontsize=22)
        # ax.text(i+0.2, f[1]-0.8, round(100*f[1]/N, 1), color='black', ha="center", fontsize=22)
        ax.text(i-0.2, f[0]-0.8, f[0], color='black', ha="center", fontsize=22)
        ax.text(i+0.2, f[1]-0.8, f[1], color='black', ha="center", fontsize=22)

    ax.xaxis.grid(False)
    ax.set_ylim([0, 30])

    ax.set(xlabel=None, ylabel=None)

    fig.savefig('./{}/binary_questions.png'.format(SAVEFOL), bbox_inches='tight')


def binary_questions_mcnemar(columns, responses_adap, responses_fixed):

    print('\n\n')

    # Evaluate binary response answers
    print('Binary questions')

    N = len(responses_adap[0])
    df = pd.DataFrame(data={})

    question_labels = ['Did the robot always \n track drill color?', 'Does the robot adapt \n to user actions?']

    for j, cidx in enumerate(binary_columns):

        print('Question: {}'.format(columns[cidx]))

        scores_adap, mode_adap = read_bernoulli(responses_adap[cidx], bool_to_score)
        scores_fixed, mode_fixed = read_bernoulli(responses_fixed[cidx], bool_to_score)

        # Contigency table row is fixed (no, yes) and column is adaptive (no, yes)
        ctable = np.zeros((2,2))

        for sa, sf in zip(scores_adap, scores_fixed):
            if sf == sa == bool_to_score['No']:
                ctable[0, 0] += 1
            elif sf == sa == bool_to_score['Yes']:
                ctable[1, 1] += 1
            elif sf == bool_to_score['No'] and sa == bool_to_score['Yes']:
                ctable[0, 1] += 1
            elif sf == bool_to_score['Yes'] and sa == bool_to_score['No']:
                ctable[1, 0] += 1

        result = mcnemar(table=ctable, exact=True)

        print('Statistic: {}'.format(result.statistic))
        print('P value for mcnemar test {:6.4f}'.format(result.pvalue))

        print('\n')


def run_comparison(columns, data, key_data):
    """ Plot survey responses from post-experiment questionnaire
    """

    print('\n\n')
    print('Run comparison questions')

    N = len(data[0])
    df = pd.DataFrame(data={})

    # Define Q labels and dictionaries
    question_labels = ['Was better at \n tracking color', 'Made the user \n wait the least', 'Was most preferred']
    response_to_legend = {'A': 'Adaptive Behavior', 'F': 'Fixed Behavior', 'Both equally': 'Both behaviors equally', 'None': 'None of the behaviors'}

    colors = {
        'Adaptive Behavior': '#74add1', 'Fixed Behavior': '#fe9929', 
        'Both behaviors equally': '#7fc97f', 'None of the behaviors': '#e78ac3'
    }

    frequencies = []

    labels = key_data[1]
    for k, cidx in enumerate(run_comp_columns):

        print('Question: {}'.format(columns[cidx]))

        frequency_data = []
        for j, row in enumerate(data[cidx]):

            choices = ['A', 'F']
            translator = {'Behavior 1': labels[j], 'Both equally': 'Both equally', 'None': 'None'}
            choices.remove(labels[j])
            translator.update({'Behavior 2': choices[0]})

            response = translator[row]

            frequency_data.append(response_to_legend[response])

        mode, frequency = mode_fn(frequency_data)
        print('Mode response: {}, frequency {}, percentage {:3.1f}'.format(mode, frequency, 100*frequency[0]/N))

        d = pd.DataFrame([[frequency_data[i], question_labels[k]] for i in range(N)],
                            columns = ['Response', 'Question'])

        # Convert Response data to ordinal 
        d['Response'] = pd.Categorical(d['Response'], 
                                       ['None of the behaviors', 'Both behaviors equally', 'Fixed Behavior', 'Adaptive Behavior'], 
                                       ordered=True)
        d.sort_values('Response')

        df = df.append(d, ignore_index=True)

        # Measure frequencies of each response for each question
        fq = d['Response'].value_counts(sort=False)[::-1]
        frequencies.append(fq)

        print('Resp percentages {}'.format([np.round(100*vc/N, 1) for vc in fq]))

        print('\n')


    fig = plt.figure(figsize=(12, 12))

    sns.set(font='Times New Roman')
    sns.set(font_scale=2)

    sns.set_style('whitegrid', {"grid.color":"0.6", "axes.edgecolor":"0.6"})

    ax = sns.histplot(data=df, x='Question', hue='Response', stat='count', multiple='stack', 
                        shrink=.6, palette=colors, legend=False, edgecolor="none", alpha=1.0)

    # Write percentages for each Q response
    for i, fq in enumerate(frequencies):
        y = 0
        for f in fq:
            if f:
                y += f
                # ax.text(i, y - 0.8, round(100*f/N, 1), color='black', ha="center", fontsize=22)
                ax.text(i, y - 0.8, f, color='black', ha="center", fontsize=22)

    # Remove x axis grid lines
    ax.xaxis.grid(False)

    # ax.set(title='User Comparison of Robot Behaviors:')
    ax.set(xlabel=None, ylabel=None)
    ax.set_ylim([0, 30])

    ## To show frequence in multiples of x
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

    '''
    # Move legend outside figure
    sns.move_legend(ax, "lower center", bbox_to_anchor=(1.15, 0.8))

    # for legend text
    plt.setp(ax.get_legend().get_texts(), fontsize='14') 
    
    # for legend title
    plt.setp(ax.get_legend().get_title(), fontsize='15') 
    '''

    fig.savefig('./{}/run_comparison.png'.format(SAVEFOL), bbox_inches='tight')


def main():
    # sns.set_style("whitegrid")

    input_folder = 'user_results'

    demographic_file_path = './{}/Demographics and Robot-experience (Responses).ods'.format(input_folder)
    adap_file_path = './{}/HRC experiment- Autorack (Responses).ods'.format(input_folder)
    fixed_file_path = './{}/HRC experiment- Flatcar (Responses).ods'.format(input_folder)
    run_comp_path = './{}/Run Comparison form (Responses).ods'.format(input_folder)

    # Gather data column wise
    _, key_data = ods_to_data('./{}/participant_key.ods'.format(input_folder), response_filter)

    columns_dgp, data_dgp = ods_to_data(demographic_file_path, response_filter=response_filter)
    columns_adap, data_adap = ods_to_data(adap_file_path, response_filter=response_filter)
    columns_fixed, data_fixed = ods_to_data(fixed_file_path, response_filter=response_filter)
    columns_comp, data_comp = ods_to_data(run_comp_path, response_filter=response_filter)
    
    # Demographics
    demographics(columns_dgp, data_dgp)

    if ARGS.binary:
        # Plot binary responses
        binary_questions(columns_adap, data_adap, data_fixed)
        # Mcnemar Test results
        binary_questions_mcnemar(columns_adap, data_adap, data_fixed)

    if ARGS.likert:
        sns.set_theme(font_scale=1.5)
        survey_questions(columns_adap, data_adap, data_fixed)

    if ARGS.tlx:
        sns.set_theme(font_scale=1.5)
        # Plot barplot for tlx score
        tlx_score(columns_adap, data_adap, data_fixed)

    # Plot post-experiment responses
    if ARGS.post_expt:
        sns.set_theme(font_scale=1.5)
        run_comparison(columns_comp, data_comp, key_data)

    print("Done")

if __name__ == '__main__':
    main()
