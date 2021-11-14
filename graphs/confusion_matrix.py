from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv


# MSE
f = open('mse.csv', 'r')
file = csv.DictReader(f, delimiter=';')

actual = []
prediction = []

for col in file:
    col.keys()
    actual.append(col['value'])
    if (float(col['MSE']) < 2000):
        prediction.append('1')
    else:
        prediction.append('0')

target_names = ['equal 1', 'unequal 0']

print(actual)
print(prediction)

print(classification_report(actual, prediction, target_names=target_names))

cm = confusion_matrix(actual, prediction, labels=['1', '0'])
print(cm)

cm_obj = ConfusionMatrixDisplay(cm, display_labels=['1', '0'])

cm_obj.plot()
cm_obj.ax_.set(
    title='MSE confusion matrix',
    xlabel='Computed',
    ylabel='Actual')
plt.show()
