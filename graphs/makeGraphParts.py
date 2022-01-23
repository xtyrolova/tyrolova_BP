
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv


# MSE
def mse(image1, image2):

    image1 = cv2.resize(image1, (1206, 1820))
    image2 = cv2.resize(image2, (1206, 1820))
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])

    return err


def makeParts(path1, path2):
    image1 = cv2.imread(path1, 1)
    image2 = cv2.imread(path2, 1)

    grayImage1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    grayImage2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    sizeX1 = image1.shape[1]
    sizeY1 = image1.shape[0]

    sizeX2 = image2.shape[1]
    sizeY2 = image2.shape[0]

    parts = 2
    f, ax = plt.subplots(parts, parts*2, figsize=(10, 5), num="Parts")
    f.suptitle(mse(grayImage1, grayImage2))
    for i in range(0, parts):
        # row = []
        for j in range(0, parts):
            start_y1 = i * sizeY1 / parts
            end_y1 = i * sizeY1 / parts + sizeY1 / parts
            start_x1 = j * sizeX1 / parts
            end_x1 = j * sizeX1 / parts + sizeX1 / parts
            roi1 = grayImage1[int(start_y1):int(end_y1), int(start_x1):int(end_x1)]
            ax[i][j].imshow(roi1)
            ax[i][j].axis('off')   # hide the axis

            start_y2 = i * sizeY2 / parts
            end_y2 = i * sizeY2 / parts + sizeY2 / parts
            start_x2 = j * sizeX2 / parts
            end_x2 = j * sizeX2 / parts + sizeX2 / parts
            roi2 = grayImage2[int(start_y2):int(end_y2), int(start_x2):int(end_x2)]

            ax[i][j+2].imshow(roi2)
            ax[i][j+2].axis('off')   # hide the axis
            ax[i][j].title.set_text(mse(roi1, roi2))
            f.tight_layout()
            print(mse(roi1, roi2))
    plt.show()


# OPEN AND LOAD DATASET
f = open('../networks/distance_metrics/dtb.csv', 'r')
file = csv.DictReader(f, delimiter=';')

ids = []
images1 = []
images2 = []
values = []

for col in file:
    col.keys()
    ids.append(col['ID'])
    images1.append(col['path1'])
    images2.append(col['path2'])
    values.append(col['value'])


makeParts(images1[16], images2[16])


# fig, axes = plt.subplots(2, 2, figsize=(10, 5))
# fig.suptitle('MSE0 vs MSE1')
#
# # maska MSE 0 a 1
# dataMSE = pd.read_csv('mse.csv', delimiter=';')
# # rozdelit do dvoch dataframov, v jednom budu len 0, v druhom len rovnake 1
# maskaM = (dataMSE.value == 0)
# dataMSE0 = dataMSE[maskaM]
# dataMSE1 = dataMSE[~maskaM]
# sns.histplot(data=dataMSE0, x='MSE', bins=20, color="pink", ax=axes[0])
#
# sns.histplot(data=dataMSE1, x='MSE', bins=20, color="green", ax=axes[1])
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# fig.suptitle('SSIM0 vs SSIM1')
