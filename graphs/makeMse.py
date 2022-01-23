import csv
import cv2
import numpy as np


def mse(imageA, imageB):

    imageA = cv2.resize(imageA, (1206, 1820))
    imageB = cv2.resize(imageB, (1206, 1820))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


# OPEN AND LOAD DATASET
f = open('../networks/distance_metrics/dtb.csv', 'r')
file = csv.DictReader(f, delimiter=';')

ids = []
images1 = []
images2 = []
values = []
id = 0

for col in file:
    col.keys()
    ids.append(id)
    images1.append(col['path1'])
    images2.append(col['path2'])
    values.append(col['value'])
    id += 1

# CREATE AND MAKE MSE
header = ['ID', 'value', 'MSE']

fileMSE = open('mse.csv', 'w+', newline='')
writer = csv.writer(fileMSE, delimiter=';')
writer.writerow(header)

for i in range(0, len(ids)):
    image1 = cv2.imread("../networks/distance_metrics/" + images1[i], cv2.COLOR_RGB2BGRA)
    image2 = cv2.imread("../networks/distance_metrics/" + images2[i], cv2.COLOR_RGB2BGRA)
    row = [ids[i], values[i], mse(image1, image2)]
    writer.writerow(row)
