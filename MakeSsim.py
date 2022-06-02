from skimage.metrics import structural_similarity as ssim
import cv2
import csv


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
header = ['ID', 'value', 'SSIM']

fileMSE = open('ssim.csv', 'w+', newline='')
writer = csv.writer(fileMSE, delimiter=';')
writer.writerow(header)

for i in range(0, len(ids)):
    image1 = cv2.imread("../networks/distance_metrics/" + images1[i], cv2.COLOR_RGB2BGRA)
    image2 = cv2.imread("../networks/distance_metrics/" + images2[i], cv2.COLOR_RGB2BGRA)
    image1 = cv2.resize(image1, (1206, 1820))
    image2 = cv2.resize(image2, (1206, 1820))
    row = [ids[i], values[i], ssim(image1, image2, multichannel=True)]
    writer.writerow(row)
