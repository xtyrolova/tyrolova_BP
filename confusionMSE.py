from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd


def find_treshold_mse(min, max, actual, prediction, mse):
    i = min
    best_acc = 0
    best_val = min
    while i < max:
        for j in range(len(mse)):
            if float(mse[j]) < i:
                prediction[j] = '1'
            else:
                prediction[j] = '0'
        acc = accuracy_score(actual, prediction, normalize=False)
        if acc > best_acc:
            best_acc = acc
            best_val = i
        i += 1.0

    return best_val


target_names = ['equal 1', 'unequal 0']

# MSE
print('MSE')
f_mse = open('mse.csv', 'r')
file_mse = csv.DictReader(f_mse, delimiter=';')

actual_mse = []
prediction_mse = []
actual_ones = []
actual_zeros = []
mse_values = []

for col in file_mse:
    col.keys()
    actual_mse.append(col['value'])
    mse_values.append(col['MSE'])
    if float(col['MSE']) < 2000:
        prediction_mse.append('1')
    else:
        prediction_mse.append('0')

ids = []
number = 0
for i in range(len(actual_mse)):
    ids.append(str(number))
    number += 1
    if actual_mse[i] == '1':
        actual_ones.append(float(mse_values[i]))
    else:
        actual_zeros.append(float(mse_values[i]))


ones_train, one_test, zeros_train, zeros_test = train_test_split(actual_ones, actual_zeros, test_size=0.2)
actual_train, actual_test, prediction_train, prediction_test = train_test_split(actual_mse, prediction_mse,
                                                                                test_size=0.2)
mse_train, mse_test = train_test_split(mse_values, test_size=0.2)

max_ones = max(ones_train)
min_ones = min(ones_train)
max_zeros = max(zeros_train)
min_zeros = min(zeros_train)
values = [float(max_ones), float(min_ones), float(max_zeros), float(min_zeros)]
max_value = max(values)
min_value = min(values)

# print(max_ones, min_ones, max_zeros, min_zeros)
# print(ids)
# print(actual_mse)
# print(prediction_mse)
tr = find_treshold_mse(min_value, max_value, actual_train, prediction_train, mse_train)
print(tr)

for i in range(len(mse_test)):
    if float(mse_test[i]) < tr:
        prediction_test[i] = '1'
    else:
        prediction_test[i] = '0'




# for i in range(len(ids)):
#     if actual_mse[i] == '1' and prediction_mse[i] == '0':
#         print("FP: ", i)
#     if actual_mse[i] == '0' and prediction_mse[i] == '1':
#         print("FN: ", i)
# print(ids)
# print(actual_mse)
# print(prediction_mse)


print(classification_report(actual_test, prediction_test, target_names=target_names))
cm = confusion_matrix(actual_test, prediction_test, labels=['1', '0'])

# print(classification_report(x_test, y_test, target_names=target_names))
# cm = confusion_matrix(x_test, y_test, labels=['1', '0'])

print(cm)

df_cm = pd.DataFrame(cm, index=[i for i in "10"],
                     columns=[i for i in "10"])
plt.figure(figsize=(10, 7))
# plt.title('Konfúzna matica pre MSE')

sn.heatmap(df_cm, annot=True, cmap="OrRd")

plt.xlabel('Predpoveď', fontsize=14)
plt.ylabel('Skutočnosť', fontsize=14)

plt.show()
