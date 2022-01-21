import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import csv


def vgg16_process(path):
    model = VGG16(weights='imagenet', include_top=False)

    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features


f = open('dtb.csv', 'r')
file = csv.DictReader(f, delimiter=';')

ids = []
values = []
images1 = []
images2 = []

for col in file:
    col.keys()
    ids.append(col['ID'])
    values.append(col['value'])
    images1.append(col['path1'])
    images2.append(col['path2'])

header = ['ID', 'value', 'Euclid']

fileMSE = open('euclid_vgg16.csv', 'w+', newline='')
writer = csv.writer(fileMSE, delimiter=';')
writer.writerow(header)

for i in range(len(ids)):
    euclid_array1 = vgg16_process(images1[i])
    euclid_array2 = vgg16_process(images2[i])

    distance = euclidean_distances(euclid_array1, euclid_array2)
    distance = np.concatenate(distance)

    row = [ids[i], values[i], distance[0]]
    writer.writerow(row)
