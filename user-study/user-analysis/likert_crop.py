"""Crop likert scale bar plots by question."""

import cv2


def crop_likert():

    # User defined bounding box to crop diagram
    xrange = [144, 263]
    yrange = [13, 1249]
    
    for i, imgname in enumerate(['./plots/likert_q{}.png'.format(i) for i in range(1, 5)]):

        img = cv2.imread(imgname)

        img = img[yrange[0]:yrange[1], xrange[0]:xrange[1], :]

        cv2.imwrite('./plots/likert_q{}_crop.png'.format(i+1), img)

    # Crop empty lines to use in diagram
    img = cv2.imread(imgname)

    img = img[yrange[0]:yrange[1], xrange[1]:(2*xrange[1]-xrange[0]), :]

    cv2.imwrite('./plots/likert_blank_crop.png'.format(i+1), img)


def crop_likert_bygroup():

    # User defined bounding box to crop diagram
    xrange = [172, 282]
    yrange = [17, 1256]

    for group in ['UHTP_TO_FIXED', 'FIXED_TO_UHTP']:

        for i, imgname in enumerate(['./plots/{}/likert_q{}_bygroup_{}.png'.format(group, i, group) for i in range(1, 5)]):

            img = cv2.imread(imgname)

            img = img[yrange[0]:yrange[1], xrange[0]:xrange[1], :]

            cv2.imwrite('./plots/{}/likert_q{}_bygroup_{}crop.png'.format(group, i+1, group), img)

    # Crop empty lines to use in diagram
    img = cv2.imread(imgname)

    img = img[yrange[0]:yrange[1], xrange[1]:(2*xrange[1]-xrange[0]), :]

    cv2.imwrite('./plots/likert_blank_crop_bygroup.png'.format(i+1), img)


def main():
    # crop_likert()

    crop_likert_bygroup()

    return

if __name__ == '__main__':
    main()
