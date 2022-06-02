import cv2
import numpy as np
import matplotlib.pyplot as plt

#import argparse
def load_image(image):

    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    #args = vars(ap.parse_args())

    print("loading image")

def show_img_comp(image1, image2):
    f, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

def average(image):
    img_temp = image.copy()
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(image, axis=(0,1))
    return img_temp

def main():
    image1 = cv2.imread("../networks/distance_metrics/images/SK_MRS_1266_A38_r.jpg", 1)
    image2 = cv2.imread("../networks/distance_metrics/images/SK_MRS_1266_B9_r.jpg", 1)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    dim = (1280, 900)
    image1 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)

    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    dim = (1280, 900)
    image2 = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)

    av_color_im1 = average(image1)
    av_color_im2 = average(image2)

    show_img_comp(image1, av_color_im1)
    show_img_comp(image2, av_color_im2)

main()