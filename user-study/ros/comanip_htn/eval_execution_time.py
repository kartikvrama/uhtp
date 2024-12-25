import csv
from math import sqrt
import numpy as np

from scipy.stats import mannwhitneyu

def generate_lines_that_equal(string, fp):
    for line in fp:
        if string in line:
            yield line


def calculate_cohens_d(mx, stdx, my, stdy):
    val = (mx - my)/sqrt(1.0*(stdx**2 + stdy**2)/2)
    return val


if __name__ == '__main__':
    users = ['n6', 'p1', 'p3', 'p4', 'p6', 'p7', 'p8', 'p10', 'p12', 'p13', 'p14', 'p15', 
            'p16', 'p17', 'p18', 'p20', 'p21']
    modes = ['adaptive', 'fixed']

    opfile = './expt_logs/execution_time_summary.csv'
    with open(opfile, 'w') as opfile:
        data = []
        writer = csv.writer(opfile, delimiter=',')
        writer.writerow(['User'] + ['Adaptive'] + ['Fixed'])
        for user in users:
            times = []
            for mode in modes:
                ipfile = './expt_logs/user_{}/User-{}_mode-{}_exop.txt'.format(user, user, mode)
                with open(ipfile, 'r') as fp:
                    for line in generate_lines_that_equal('Total time taken in seconds:', fp):
                        ext = line
                        break
                    t = float(ext[-7:])
                    times.append(t)
            writer.writerow([user] + times)
            data.append(times)

    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print(mannwhitneyu(data[:, 0], data[:, 1]))
    print(calculate_cohens_d(mean[0], std[0], mean[1], std[1]))