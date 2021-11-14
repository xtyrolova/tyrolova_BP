from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv

target_names = ['equal 1', 'unequal 0']

# MSE
print('MSE')
f_mse = open('mse.csv', 'r')
file_mse = csv.DictReader(f_mse, delimiter=';')

actual_mse = []
prediction_mse = []

for col in file_mse:
    col.keys()
    actual_mse.append(col['value'])
    if (float(col['MSE']) < 2000):
        prediction_mse.append('1')
    else:
        prediction_mse.append('0')

print(actual_mse)
print(prediction_mse)

print(classification_report(actual_mse, prediction_mse, target_names=target_names))

cm = confusion_matrix(actual_mse, prediction_mse, labels=['1', '0'])
print(cm)

cm_obj = ConfusionMatrixDisplay(cm, display_labels=['1', '0'])

cm_obj.plot()
cm_obj.ax_.set(
    title='MSE confusion matrix',
    xlabel='Computed',
    ylabel='Actual')
plt.show()


# SSIM
print('SSIM')
f_ssim = open('ssim.csv', 'r')
file_ssim = csv.DictReader(f_ssim, delimiter=';')

actual_ssim = []
prediction_ssim = []

for col in file_ssim:
    col.keys()
    actual_ssim.append(col['value'])
    if (float(col['SSIM']) > 0.45):
        prediction_ssim.append('1')
    else:
        prediction_ssim.append('0')

print(actual_ssim)
print(prediction_ssim)

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
