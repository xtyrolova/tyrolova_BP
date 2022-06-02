import os
from random import random
import random
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly
import os
import numpy as np

from matplotlib.pyplot import hist
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import time
import numpy as np


sns.set_palette("Set2")
data = pd.read_csv("../keras/vgg16.csv", delimiter=",")
data.head()

# --------------
# RANDOM FOREST
# --------------

rf = RandomForestClassifier()
x, y = data.drop(['0', '513'], axis=1), data['513']       # vgg16
# x, y = data.drop(['0', '2049'], axis=1), data['2049']       # resnet50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
rf.fit(x_train, y_train)
prediction = rf.predict(x_test)

print('With Random Forest accuracy is: ', rf.score(x_test, y_test))     # accuracy

# feature importances

plt.rcParams.update({'figure.figsize': (10.0, 7.0)})
plt.rcParams.update({'font.size': 14})

# zeroless_importances = [i for i in rf.feature_importances_ if i != 0.]
zeroless_importances = rf.feature_importances_[rf.feature_importances_ != 0.0]
# print(zeroless_importances)

ind = np.argpartition(rf.feature_importances_, -10)[-10:]
top10 = rf.feature_importances_[ind]

ind = np.argpartition(zeroless_importances, -10)[:10]
worst10 = zeroless_importances[ind]

# print(top10)
# print(worst10)
features = np.concatenate((worst10, top10))
# print(features)
feature_names = [f"príznak {i}" for i in range(len(worst10)+len(top10))]
plt.barh(feature_names, features)

from decimal import Decimal

for index, value in enumerate(features):
    plt.text(value, index,
             str(round(Decimal(value), 4)))

cm = confusion_matrix(y_test, prediction)
print('Confusion matrix\n\n', cm)

df_cm = pd.DataFrame(cm, index=[i for i in "10"],
                     columns=[i for i in "10"])
plt.figure(figsize=(10, 7))

sns.heatmap(df_cm, annot=True, cmap="OrRd")

plt.xlabel('Predpoveď', fontsize=14)
plt.ylabel('Skutočnosť', fontsize=14)
plt.show()


# directory = 'images'  #ulozene obrazky, ktore chcem zobrazit
#
# m = 10
# chosens = random.sample(range(1, 2050), m) #nahodny stlpec zo stlpcov kde su priznaky
# fig = make_subplots(cols=2, rows=m,subplot_titles=([val for val in chosens for _ in (0, 1)]))
#
# for i in range (m):
#     column = data.columns[chosens[i]] #prilozim screenshot
#     # plot histogram
#     fig.add_trace(go.Histogram(x=data[column], histnorm='probability'),row=i+1,col=1)
#     view = data.sort_values(by=column,ascending=False).head(n=200) #zorad dataframe podla hodnoty priznaku a nechaj si prvych 200 riadkov
#     imgs = []
#     for index,row in view.iterrows():
#         filename = row['filename']
#         full_path = os.path.join(directory, filename)
#
#         img = dlib.load_rgb_image(full_path) #ak mame velke obrazky tu je ich treba zmensit
#         imgs.append (img)
#
#     tiled_images = dlib.tile_images(imgs[0:100])
#     tiled_images2 = dlib.tile_images(imgs[100:200])
#     tiled_final = np.hstack((tiled_images,tiled_images2))
#     fig.add_trace(go.Image(z=tiled_final),row=i+1,col=2)
#     # fig.add_trace(go.Image(z=tiled_images2),row=i+1,col=3)
#
# fig.update_layout(showlegend=False, height = m*700)
# fig.write_html ('priznaky.html')
