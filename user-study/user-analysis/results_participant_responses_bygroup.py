import argparse
import numpy as np
from math import sqrt
from copy import deepcopy

from scipy.stats import mode as mode_fn, mannwhitneyu, chisquare
from utils_results import read_bernoulli, ods_to_data, read_4pt_likert, sns_change_width

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# As read in participant_key row number

adap_first_filter = [5, 9, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
fixed_first_filter = [6, 8, 12, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]

bool_to_score = {'Yes': 1, 'No': 2}
likert_to_score = {'Always': 4, 'Most of the Time': 3, 'Sometimes': 2, 'Never': 1}
reverselikert_to_score = {'Always': 1, 'Most of the Time': 2, 'Sometimes': 3, 'Never': 4}

likert_columns = [2, 3, 4, 5]
tlx_columns = [7, 8, 9, 10, 11, 12]

run_comp_columns = [1, 2, 3]

SAVEFOL = 'plots'

parser=argparse.ArgumentParser()
parser.add_argument("--survey", type=bool, default=False, help="Whether to analyze survey responses")
parser.add_argument("--tlx", type=bool, default=False, help="Whether to analyze NASA TLX responses")
ARGS = parser.parse_args()

def compare_tlx_questions(columns, responses_first, responses_second, savelabel='', xlabels=['First', 'Second']):
    N = len(responses_first[0])

    df = pd.DataFrame(data={})

    print('\n\n')

    scores_1 = np.array([responses_first[i] for i in tlx_columns])
    scores_1 = np.sum(scores_1, axis=0)

    scores_2 = np.array([responses_second[i] for i in tlx_columns])
    scores_2 = np.sum(scores_2, axis=0)

    print('{} ... Median {} ... Min {} ... Max {} ...'.format(xlabels[0], np.median(scores_1), np.min(scores_1), np.max(scores_1)))
    print('{} ... Median {} ... Min {} ... Max {} ...'.format(xlabels[1], np.median(scores_2), np.min(scores_2), np.max(scores_2)))

    d1 = pd.DataFrame([[scores_1[i], xlabels[0], ''] for i in range(N)],
                            columns = ['Data', 'Robot Behavior', 'Question'])

    d2 = pd.DataFrame([[scores_2[i], xlabels[1], ''] for i in range(N)],
                            columns = ['Data', 'Robot Behavior', 'Question'])

    df = pd.concat([d1, d2], ignore_index=True)

    fig = plt.figure(figsize=(8, 8))
  
    colors = {'SUHTP': '#74add1', # uhtp
            'SFIXED': '#fe9929' # fixed
            }

    curr_pallette = [colors[k] for k in xlabels]

    fig = plt.figure(figsize=(16, 9))

    sns.set(font='Times New Roman')
    sns.set(font_scale=3)
    
    sns.set_style('whitegrid', {"grid.color":"0.1", "axes.edgecolor":"0.1"})

    # Bar plot
    ax = sns.boxplot(data=df, y='Question', x='Data', hue='Robot Behavior', 
                     orient='h', palette=colors)

    ax.set(xlim=[0, 24])
    ax.set_xticks(range(0, 25, 2))
    ax.set(ylabel=None, xlabel="NASA TLX Score (lesser is better)")
    ax.get_legend().remove()

    fig.savefig('./{}/group_{}_tlx.png'.format(SAVEFOL, savelabel), bbox_inches='tight')


def compare_survey_questions(columns, responses_first, responses_second, savelabel='', xlabels=['First', 'Second']):

    # Evaluate Likert scale answers
    print('Likert items')

    N = len(responses_first[0])

    question_labels = ['RAcc*', 'RFR', 'RID', 'RIA']
    file_names = ['q1', 'q2', 'q3', 'q4']

    comparisons = ['less', 'less', 'less', 'less']
    answer_count_o1 = {}
    answer_count_o2 = {}

    for j, cidx in enumerate(likert_columns):
        answer_count_o1[question_labels[j]] = {i:0 for i in range(1, 5)}
        answer_count_o2[question_labels[j]] = {i:0 for i in range(1, 5)}

        if j == 0: # RAcc*
            scores_first, median_first, binc_first = read_4pt_likert(responses_first[cidx], reverselikert_to_score)
            scores_second, median_second, binc_second = read_4pt_likert(responses_second[cidx], reverselikert_to_score)

        else:
            scores_first, median_first, binc_first = read_4pt_likert(responses_first[cidx], likert_to_score)
            scores_second, median_second, binc_second = read_4pt_likert(responses_second[cidx], likert_to_score)

        # Collecting bin counts in 
        for key in binc_first:
            answer_count_o1[question_labels[j]][key] = binc_first[key]

        for key in binc_second:
            answer_count_o2[question_labels[j]][key] = binc_second[key]

    responses = ['Never', 'Sometimes', 'Most of the time', 'Always']

    width = 0.075
    x = [0.2, 0.2+width] #['Adaptive\nHTN', 'Fixed\nSequence']

    for idx, q in enumerate(question_labels):

        num_participants = 16

        # Centering adaptive and fixed behavior data b/w positive and negative responses

        data_o1 = [answer_count_o1[q][k] for k in range(1, 5)]
        data_o1.insert(0, num_participants - data_o1[0] - data_o1[1])
        data_o1.insert(5, num_participants - data_o1[0])

        data_o2 = [answer_count_o2[q][k] for k in range(1, 5)]
        data_o2.insert(0, num_participants - data_o2[0] - data_o2[1])
        data_o2.insert(5, num_participants - data_o2[0])

        ## Plot Adaptive v/s Fixed frequency data
        y_data = np.vstack([data_o1, data_o2]).T

        # Bottom buffer
        y_prev = y_data[0, :]
        plt.bar(x, y_prev, label='1', alpha=0.0, width=width)

        colors = {
                'SUHTP':[(189,201,225),
                      (116,169,207),
                      (43,140,190),
                      (4,90,141)],
                'SFIXED':[(254,217,142),
                       (254,153,41),
                       (217,95,14),
                       (153,52,4)]
        } 

        alphas = [1, 1, 1, 1]

        fig=plt.figure(figsize=(9, 16))

        sns.set_style('whitegrid', {"grid.color":"0", "axes.edgecolor":"0"})

        axes = plt.gca()
        axes.xaxis.grid(False)

        # Stacking bar plots on top of previous bar
        for i in range(1, len(data_o1)-1):
            y_curr = y_data[i, :]

            color_o1 = colors[xlabels[0]][i-1]        
            color_o2 = colors[xlabels[1]][i-1]        

            axes.bar(x[0], y_curr[0], bottom=y_prev[0], label=responses[i-1], 
                        color=[c/255.0 for c in color_o1], alpha=alphas[i-1], 
                        width=width, edgecolor="none")

            axes.bar(x[1], y_curr[1], bottom=y_prev[1], label=responses[i-1], 
                        color=[c/255.0 for c in color_o2], alpha=alphas[i-1], 
                        width=width, edgecolor="none")

            y_prev += np.array(y_curr)

            # Top buffer
            y_curr = y_data[-1, :]
            axes.bar(x, y_curr, bottom=y_prev, label='2', alpha=0.0, width=width)

            # Axis formatting
            axes.set(ylim=[0, 2*num_participants])
            axes.set_yticks(range(0, 2*num_participants+1, 4))

            axes.set_xticks(x)
            axes.set_xticklabels([xlabels[0], xlabels[1]])

            axes.set_xlim(0, 1)

        fig.savefig('./{}/{}/likert_{}_bygroup_{}'.format(SAVEFOL, savelabel, file_names[idx], savelabel), bbox_inches='tight')

        print('')

def main():

    input_folder = 'user_results'

    demographic_file_path = './{}/Demographics and Robot-experience (Responses).ods'.format(input_folder)
    adap_file_path = './{}/HRC experiment- Autorack (Responses).ods'.format(input_folder)
    fixed_file_path = './{}/HRC experiment- Flatcar (Responses).ods'.format(input_folder)
    run_comp_path = './{}/Run Comparison form (Responses).ods'.format(input_folder)

    sns.set(font_scale=2)

    response_filter = adap_first_filter
    _, key_data = ods_to_data('./{}/participant_key.ods'.format(input_folder), response_filter)
    columns_adap, data_adap = ods_to_data(adap_file_path, response_filter=response_filter)
    columns_fixed, data_fixed = ods_to_data(fixed_file_path, response_filter=response_filter)
    if ARGS.survey:
        compare_survey_questions(columns_adap, data_adap, data_fixed, savelabel='UHTP_TO_FIXED', 
                                    xlabels=['SUHTP', 'SFIXED'])
    if ARGS.tlx:
        compare_tlx_questions(columns_adap, data_adap, data_fixed, savelabel='Adaptive to Fixed behavior', 
                                    xlabels=['SUHTP', 'SFIXED'])

    response_filter = fixed_first_filter
    _, key_data = ods_to_data('./{}/participant_key.ods'.format(input_folder), response_filter)
    columns_adap, data_adap = ods_to_data(adap_file_path, response_filter=response_filter)
    columns_fixed, data_fixed = ods_to_data(fixed_file_path, response_filter=response_filter)
    if ARGS.survey:
        compare_survey_questions(columns_adap, data_fixed, data_adap, savelabel='FIXED_TO_UHTP', 
                                    xlabels=['SFIXED', 'SUHTP'])
    if ARGS.tlx:
        compare_tlx_questions(columns_adap, data_fixed, data_adap, savelabel='Fixed to Adaptive behavior', 
                                    xlabels=['SFIXED', 'SUHTP'])

    print("Done")


if __name__ == '__main__':
    main()
