import os
import argparse

import csv
import numpy as np

PARENT_FOLDER = '/home/comanip/adapcomanip_ws/src/comanip_htn/expt_logs'


def dict2txt(filename, data):
    with open(filename, 'w') as datafile:
        for key in data.keys():
            string = '{}:- {:7.3f}\n'.format(key, data[key])
            print(string)
            datafile.write(string)


def textcsv2list(csvfile):
    data = []
    with open(csvfile, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            data.append([str(i) for i in row if i != ''])
    data = np.array(data)
    return data


def calculate_fluency(user, mode):
    folder_path = os.path.join(PARENT_FOLDER, 'user_{}'.format(user))
    csv_file = os.path.join(folder_path, '{}-{}_fluency.csv'.format(user, mode))

    frame_labels = textcsv2list(csvfile=csv_file)

    h_idle = 0
    r_idle = 0
    shared = 0
    T = 0

    for idrow, row in enumerate(frame_labels):
        if len(row) > 0:
            dt = int(row[1]) - int(row[0])
            T += dt
            label = row[-1]
            if label == 'idle':
                r_idle += dt
            elif label == 'robot':
                h_idle += dt
            elif label == 'shared':
                shared += dt
            else:
                print('Row number {} defective'.format(idrow))
                raise KeyError
    
    metric_dict = {'H-idle metric': 1.0*h_idle/T,
                   'R-idle metric': 1.0*r_idle/T,
                   'Shared time metric': 1.0*shared/T,
                   'Total time': 1.0*T/5
                   }
    return metric_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-user')
    args = parser.parse_args()

    print('Evaluating User {}'.format(args.user))

    # Calculating metrics
    fixed_mode_metrics = calculate_fluency(args.user, 'fixed')
    adap_mode_metrics = calculate_fluency(args.user, 'adaptive')

    # Save fixed mode metric file
    print('\nSaving fixed mode metrics')
    fixed_metric_file = os.path.join(PARENT_FOLDER, 'user_{}/User-{}_mode-fixed_metrics.txt' \
                            .format(args.user, args.user))
    dict2txt(fixed_metric_file, fixed_mode_metrics)

    # Save adap mode metric file
    print('\nSaving adap mode metrics')
    adap_metric_file = os.path.join(PARENT_FOLDER, 'user_{}/User-{}_mode-adap_metrics.txt' \
                            .format(args.user, args.user))
    dict2txt(adap_metric_file, adap_mode_metrics)

if __name__ == '__main__':
    main()
