import csv

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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

# median = st.median(euclid)
# print(median)

for i in range(len(actual)):
    if actual[i] == '1':
        ones.append(manhattan[i])
    else:
        zeros.append(manhattan[i])

max_ones = max(ones)
min_ones = min(ones)
max_zeros = max(zeros)
min_zeros = min(zeros)
print("max 1:", max_ones, " min 1:", min_ones)
print("max 0:", max_zeros, " min 0:", min_zeros)
values = [float(max_ones), float(min_ones), float(max_zeros), float(min_zeros)]
max_value = max(values)
min_value = min(values)

for i in range(len(manhattan)):
    if manhattan[i] < 0.5:
        prediction.append('1')
    else:
        prediction.append('0')

tr = find_treshold(min_value, max_value, actual, prediction, manhattan)
# print(actual)
# print(prediction)
print(tr)

for i in range(len(manhattan)):
    if manhattan[i] < tr:
        prediction[i] = '1'
    else:
        prediction[i] = '0'


for i in range(len(actual)):
    if actual[i] == '1' and prediction[i] == '0':
        print("FP: ", i)
    if actual[i] == '0' and prediction[i] == '1':
        print("FN: ", i)

target_names = ['equal 1', 'unequal 0']
print(classification_report(actual, prediction, target_names=target_names))
print(accuracy_score(actual, prediction, normalize=False))

cm = confusion_matrix(actual, prediction, labels=['1', '0'])

cm_obj = ConfusionMatrixDisplay(cm, display_labels=['1', '0'])

cm_obj.plot()
cm_obj.ax_.set(
    title='Manhattan confusion matrix',
    xlabel='Computed',
    ylabel='Actual')
plt.show()
