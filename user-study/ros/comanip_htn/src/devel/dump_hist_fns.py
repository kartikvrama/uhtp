
def get_hist(self, img, mask):
    hist = np.zeros((BINCOUNT, 3))
    for i in range(3):
        hist[:, i] = cv2.calcHist([img],[i],(mask*255).astype('uint8'),[BINCOUNT],[0,256]).ravel()
    return hist.astype('float32')

def calculate_avg_hist(self):
    avg_green_hist = np.zeros(BINCOUNT*3)
    avg_yellow_hist = np.zeros(BINCOUNT*3)

    for i in range(4):
        img = self.get_image()
        print('Approve this image?')
        self.plot_masked(img)

        green_hist = self.get_hist(img, self.green_mask)
        avg_green_hist += green_hist

        yellow_hist = self.get_hist(img, self.yellow_mask)
        avg_yellow_hist += yellow_hist

        raw_input('Done with {}: Next?'.format(i+1))
        print('')
    return avg_green_rgb, avg_yellow_rgb
    # cv2.compareHist(greenhist.ravel(), self.greenhist_template.ravel(), cv2.HISTCMP_CHISQR) 

    # cv2.compareHist(yellowhist.ravel(), self.yellowhist_template.ravel(), cv2.HISTCMP_CHISQR) 

