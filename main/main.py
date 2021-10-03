import cv2
#import argparse
def load_image():

    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    #args = vars(ap.parse_args())

    image = cv2.imread("C:/Users/tyrol_6l57e8e/BP_tyrolova/main/fei.jpeg")

    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("FEI STU '21", image)
    cv2.waitKey(0)

load_image()