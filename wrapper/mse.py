import cv2
import numpy as np

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    imageB = cv2.resize(imageB, (1206,1820))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

image1 = cv2.imread("SK_MRS_1264_A1_r.jpg")
image2 = cv2.imread("SK_MRS_1264_A2_r.jpg")

print(mse(image1, image2))