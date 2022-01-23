from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv


def find_treshold(min, max, actual, prediction, mse):
    i = float(min)
    best_acc = 0
    best_val = min
    while i < float(max):
        for j in range(len(mse)):
            if float(mse[j]) < i:
                prediction[j] = '1'
            else:
                prediction[j] = '0'
        acc = accuracy_score(actual, prediction, normalize=False)
        if acc > best_acc:
            best_acc = acc
            best_val = i
        i += 100.0

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
    if (float(col['MSE']) < 2000):
        prediction_mse.append('1')
    else:
        prediction_mse.append('0')

for i in range(len(actual_mse)):
    if actual_mse[i] == '1':
        actual_ones.append(mse_values[i])
    else:
        actual_zeros.append(mse_values[i])

max_ones = max(actual_ones)
min_ones = min(actual_ones)
max_zeros = max(actual_zeros)
min_zeros = min(actual_zeros)

# print(max_ones, min_ones, max_zeros, min_zeros)
# print(actual_mse)
# print(prediction_mse)
tr = find_treshold(min_ones, min_zeros, actual_mse, prediction_mse, mse_values)
print(tr)

for i in range(len(mse_values)):
    if float(mse_values[i]) < tr:
        prediction_mse[i] = '1'
    else:
        prediction_mse[i] = '0'


print(classification_report(actual_mse, prediction_mse, target_names=target_names))

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
    if (float(col['SSIM']) > 0.45):
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

tr = find_treshold(min_ones, min_zeros, actual_ssim, prediction_ssim, ssim_values)
print(tr)

for i in range(len(mse_values)):
    if mse_values[i] < tr:
        prediction_mse[i] = '1'
    else:
        prediction_mse[i] = '0'

# print(actual_ssim)
# print(prediction_ssim)

print(classification_report(actual_ssim, prediction_ssim, target_names=target_names))

cm2 = confusion_matrix(actual_ssim, prediction_ssim, labels=['1', '0'])
print(cm2)

cm_obj2 = ConfusionMatrixDisplay(cm2, display_labels=['1', '0'])

cm_obj2.plot()
cm_obj2.ax_.set(
    title='SSIM confusion matrix',
    xlabel='Computed',
    ylabel='Actual')
plt.show()
