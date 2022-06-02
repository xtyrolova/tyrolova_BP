from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

sns.set_palette("Set2")
data = pd.read_csv("../keras/vgg16.csv", delimiter=",")
data.head()

x, y = data.drop(['0', '513'], axis=1), data['513']  # vgg16
# x, y = data.drop(['0', '2049'], axis=1), data['2049']


svc = SVC(kernel="linear", C=1, gamma="auto", verbose=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = svc.fit(x_train, y_train)
prediction = svc.predict(x_test)
print('With SVM accuracy is: ', svc.score(x_test, y_test))

cm = confusion_matrix(y_test, prediction)
print('Confusion matrix\n\n', cm)

df_cm = pd.DataFrame(cm, index=[i for i in "10"],
                     columns=[i for i in "10"])
plt.figure(figsize=(10, 7))

sns.heatmap(df_cm, annot=True, cmap="OrRd")

plt.xlabel('Predpoveď', fontsize=14)
plt.ylabel('Skutočnosť', fontsize=14)
plt.show()


# X = data.iloc[:,0:2049]
# y=data['2049']
#
#
# ax = plt.gca()
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=50, cmap='autumn')
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# xx = np.linspace(xlim[0], xlim[1], 2)
# yy = np.linspace(ylim[0], ylim[1], 1024)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = model.decision_function(xy).reshape(XX.shape)
#
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
#
# ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none', edgecolors='k')
# plt.show()
