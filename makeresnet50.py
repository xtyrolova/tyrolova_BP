import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import csv

model = ResNet50(weights='imagenet', include_top=False)


def resnet50_process(path):
    img = image.load_img(path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds


# OPEN AND LOAD DATASET
f = open('../distance_metrics/dtb.csv', 'r')
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

# CREATE AND MAKE CSV

row_values = []
rows = []
for i in range(0, len(ids)):
    netImage1 = resnet50_process("../distance_metrics/" + images1[i])
    vector1 = np.reshape(netImage1, -1)
    netImage2 = resnet50_process("../distance_metrics/" + images2[i])
    vector2 = np.reshape(netImage2, -1)
    vector = (vector2 - vector1).tolist()
    vector.insert(0, i)
    vector.insert(len(vector) + 1, values[i])
    rows.append(vector)

df = pd.DataFrame(rows)
df.to_csv('resnet50.csv', index=False)
