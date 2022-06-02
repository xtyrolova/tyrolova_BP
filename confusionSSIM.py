from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd


def find_treshold_ssim(min, max, actual, prediction, ssim):
    i = min
    best_acc = 0
    best_val = min
    while i < max:
        for j in range(len(ssim)):
            if float(ssim[j]) < i:
                prediction[j] = '1'
            else:
                prediction[j] = '0'
        acc = accuracy_score(actual, prediction, normalize=False)
        if acc > best_acc:
            best_acc = acc
            best_val = i
        i += 0.001

    return best_val


target_names = ['equal 1', 'unequal 0']

# SSIM
print('SSIM')
f_ssim = open('ssim.csv', 'r')
file_ssim = csv.DictReader(f_ssim, delimiter=';')

actual_ssim = []
prediction_ssim = []
ones = []
zeros = []
ssim_values = []

for col in file_ssim:
    col.keys()
    actual_ssim.append(col['value'])
    ssim_values.append(col['SSIM'])
    if float(col['SSIM']) > 0.45:
        prediction_ssim.append('1')
    else:
        prediction_ssim.append('0')

for i in range(len(actual_ssim)):
    if actual_ssim[i] == '1':
        ones.append(ssim_values[i])
    else:
        zeros.append(ssim_values[i])

ones_train, one_test, zeros_train, zeros_test = train_test_split(ones, zeros, test_size=0.2)
actual_train, actual_test, prediction_train, prediction_test = train_test_split(actual_ssim, prediction_ssim,
                                                                                test_size=0.2)
ssim_train, ssim_test = train_test_split(ssim_values, test_size=0.2)

max_ones = max(ones_train)
min_ones = min(ones_train)
max_zeros = max(zeros_train)
min_zeros = min(zeros_train)
values = [float(max_ones), float(min_ones), float(max_zeros), float(min_zeros)]
max_value = max(values)
min_value = min(values)

# print(max_ones, min_ones, max_zeros, min_zeros)
# print(min_value, max_value)
tr = find_treshold_ssim(min_value, max_value, actual_train, prediction_train, ssim_train)
print(tr)

for i in range(len(ssim_test)):
    if float(ssim_test[i]) < tr:
        prediction_test[i] = '1'
    else:
        prediction_test[i] = '0'

# print(actual_ssim)
# print(prediction_ssim)

# print(classification_report(actual_ssim, prediction_ssim, target_names=target_names))
#
# cm2 = confusion_matrix(actual_ssim, prediction_ssim, labels=['1', '0'])
print(classification_report(actual_test, prediction_test, target_names=target_names))

cm = confusion_matrix(actual_test, prediction_test, labels=['1', '0'])
print(cm)
# for i in range(len(ids)):
#     if actual_mse[i] == '1' and prediction_ssim[i] == '0':
#         print("FP: ", i)
#     if actual_mse[i] == '0' and prediction_ssim[i] == '1':
#         print("FN: ", i)
# print(ids)
# print(actual_ssim)
# print(prediction_ssim)

df_cm = pd.DataFrame(cm, index=[i for i in "10"],
                     columns=[i for i in "10"])
plt.figure(figsize=(10, 7))
# plt.title('Konfúzna matica pre MSE')

sn.heatmap(df_cm, annot=True, cmap="OrRd")

plt.xlabel('Predpoveď', fontsize=14)
plt.ylabel('Skutočnosť', fontsize=14)

plt.show()
