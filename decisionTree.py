import os

from matplotlib import pyplot as plt
import seaborn as sn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


sns.set_palette("Set2")
data = pd.read_csv("../keras/vgg16.csv", delimiter=",")
data.head()

# ------------
# DECISION TREE
# ------------

dtree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5)
x, y = data.drop(['0', '513'], axis=1), data['513']
# x, y = data.drop(['0', '2049'], axis=1), data['2049']       # resnet50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
dtree.fit(x_train, y_train)
prediction = dtree.predict(x_test)
print(y_test)
print(prediction)

print('With Decision tree accuracy is: ', dtree.score(x_test, y_test))
target_names = ['1', '0']
from sklearn import tree
tree.export_graphviz(dtree, out_file="dec_tree_v.dot",
                     class_names=target_names,
                     rounded=True, proportion=False,
                     precision=2, filled=True)

cm = confusion_matrix(y_test, prediction)
print('Confusion matrix\n\n', cm)

df_cm = pd.DataFrame(cm, index=[i for i in "10"],
                     columns=[i for i in "10"])
plt.figure(figsize=(10, 7))

sn.heatmap(df_cm, annot=True, cmap="OrRd")

plt.xlabel('Predpoveď', fontsize=14)
plt.ylabel('Skutočnosť', fontsize=14)

plt.show()
