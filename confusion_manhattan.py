import csv

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd


def find_treshold(min, max, actual, prediction, euclid):
    i = min
    best_acc = 0
    while i < max:
        for j in range(len(euclid)):
            if euclid[j] < i:
                prediction[j] = '1'
            else:
                prediction[j] = '0'
        acc = accuracy_score(actual, prediction, normalize=False)
        if acc > best_acc:
            best_acc = acc
            best_val = i
        i += 0.1
    return best_val


f = open('manhattan_vgg16.csv', 'r')
data = csv.DictReader(f, delimiter=';')

actual = []
prediction = []
manhattan = []

ones = []
zeros = []

for col in data:
    manhattan.append(col['Manhattan'])
    actual.append(col['value'])

manhattan = list(map(float, manhattan))

for i in range(len(actual)):
    if actual[i] == '1':
        ones.append(manhattan[i])
    else:
        zeros.append(manhattan[i])

for i in range(len(manhattan)):
    if manhattan[i] < 0.5:
        prediction.append('1')
    else:
        prediction.append('0')


ones_train, one_test, zeros_train, zeros_test = train_test_split(ones, zeros, test_size=0.2)
actual_train, actual_test, prediction_train, prediction_test = train_test_split(actual, prediction,
                                                                                test_size=0.2)
x_train, y_test = train_test_split(manhattan, test_size=0.2)

max_ones = max(ones_train)
min_ones = min(ones_train)
max_zeros = max(zeros_train)
min_zeros = min(zeros_train)
values = [float(max_ones), float(min_ones), float(max_zeros), float(min_zeros)]
max_value = max(values)
min_value = min(values)

tr = find_treshold(min_value, max_value, actual_train, prediction_train, x_train)
print(tr)

for i in range(len(y_test)):
    if y_test[i] < tr:
        prediction_test[i] = '1'
    else:
        prediction_test[i] = '0'


for i in range(len(y_test)):
    if y_test[i] == '1' and prediction[i] == '0':
        print("FP: ", i)
    if y_test[i] == '0' and prediction[i] == '1':
        print("FN: ", i)

target_names = ['equal 1', 'unequal 0']
print(classification_report(actual_test, prediction_test, target_names=target_names))
print(accuracy_score(actual_test, prediction_test, normalize=False))

cm = confusion_matrix(actual_test, prediction_test, labels=['1', '0'])
print(cm)

df_cm = pd.DataFrame(cm, index=[i for i in "10"],
                     columns=[i for i in "10"])
plt.figure(figsize=(10, 7))

sn.heatmap(df_cm, annot=True, cmap="OrRd")

plt.xlabel('Predpoveď', fontsize=14)
plt.ylabel('Skutočnosť', fontsize=14)

plt.show()
