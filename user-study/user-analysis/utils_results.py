from math import sqrt

import itertools
import numpy as np

from scipy.stats import mode as mode_fn
from pyexcel_ods import get_data


def sns_change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def argsort_list(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def ods_to_data(path, response_filter):
    odict = get_data(path)
    key = list(odict.keys())[0]

    columns = odict[key][0]

    allrows = odict[key][1:]
    l = [allrows[i-1] for i in response_filter] # -1 because python index starts from 0
    data = list(map(list, itertools.zip_longest(*l, fillvalue=None)))

    return columns, data


def calculate_cohens_d(mx, stdx, my, stdy):
    val = (mx - my)/sqrt(1.0*(stdx**2 + stdy**2)/2)
    return val

def read_bernoulli(data, bool_to_score):
    new_data = [bool_to_score[d] for d in data]

    mode_value = mode_fn(new_data)[0]

    mode_response = []

    for m in mode_value:
        score_to_bool = {bool_to_score[t]: t for t in bool_to_score}
        mode_response.append(score_to_bool[m])

    if len(mode_response) == 1:
        mode_response = mode_response[0]

    return new_data, mode_response


def read_4pt_likert(data, string_to_score=None):
    # Read Likert reponses
    new_data = []
    bincount = {}

    for d in data:
        if d is not None:

            if string_to_score is None:
                item = int(d)

            else:
                item = string_to_score[d]

            new_data.append(item)

            if item in bincount.keys():
                bincount[item] += 1

            else:
                bincount[item] = 1

    median_value = np.median(new_data)

    if string_to_score is None:
        return new_data, median_value, bincount

    else:
        score_to_string = {string_to_score[t]: t for t in string_to_score}

        try: 
            median_response = score_to_string[median_value]

        except KeyError as e:
            print('Note: median {} is not an integer'.format(e))
            median_response = median_value

        return new_data, median_response, bincount