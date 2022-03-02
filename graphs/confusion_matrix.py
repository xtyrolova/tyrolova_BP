from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv


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

max_ones = max(actual_ones)
min_ones = min(actual_ones)
max_zeros = max(actual_zeros)
min_zeros = min(actual_zeros)
values = [float(max_ones), float(min_ones), float(max_zeros), float(min_zeros)]
max_value = max(values)
min_value = min(values)

# print(max_ones, min_ones, max_zeros, min_zeros)
# print(ids)
# print(actual_mse)
# print(prediction_mse)
tr = find_treshold_mse(min_value, max_value, actual_mse, prediction_mse, mse_values)
print(tr)

for i in range(len(mse_values)):
    if float(mse_values[i]) < tr:
        prediction_mse[i] = '1'
    else:
        prediction_mse[i] = '0'


print(classification_report(actual_mse, prediction_mse, target_names=target_names))

for i in range(len(ids)):
    if actual_mse[i] == '1' and prediction_mse[i] == '0':
        print("FP: ", i)
    if actual_mse[i] == '0' and prediction_mse[i] == '1':
        print("FN: ", i)
# print(ids)
# print(actual_mse)
# print(prediction_mse)

cm = confusion_matrix(actual_mse, prediction_mse, labels=['1', '0'])
print(cm)

cm_obj = ConfusionMatrixDisplay(cm, display_labels=['1', '0'])

cm_obj.plot()
cm_obj.ax_.set(
    title='MSE confusion matrix',
    xlabel='Computed',
    ylabel='Actual')
# plt.show()


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

max_ones = max(ones)
min_ones = min(ones)
max_zeros = max(zeros)
min_zeros = min(zeros)
values = [float(max_ones), float(min_ones), float(max_zeros), float(min_zeros)]
max_value = max(values)
min_value = min(values)

# print(max_ones, min_ones, max_zeros, min_zeros)
# print(min_value, max_value)
tr = find_treshold_ssim(min_value, max_value, actual_ssim, prediction_ssim, ssim_values)
print(tr)

for i in range(len(ssim_values)):
    if float(ssim_values[i]) < tr:
        prediction_ssim[i] = '1'
    else:
        prediction_ssim[i] = '0'

# print(actual_ssim)
# print(prediction_ssim)

print(classification_report(actual_ssim, prediction_ssim, target_names=target_names))

cm2 = confusion_matrix(actual_ssim, prediction_ssim, labels=['1', '0'])
print(cm2)
for i in range(len(ids)):
    if actual_mse[i] == '1' and prediction_ssim[i] == '0':
        print("FP: ", i)
    if actual_mse[i] == '0' and prediction_ssim[i] == '1':
        print("FN: ", i)
# print(ids)
# print(actual_ssim)
# print(prediction_ssim)
cm_obj2 = ConfusionMatrixDisplay(cm2, display_labels=['1', '0'])

cm_obj2.plot()
cm_obj2.ax_.set(
    title='SSIM confusion matrix',
    xlabel='Computed',
    ylabel='Actual')
plt.show()
