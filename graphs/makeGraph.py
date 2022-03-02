import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# class Photo:
#     def __init__(self, id, first, second, value):
#         self.id = id
#         self.first = first
#         self.second = second
#         self.value = value

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('MSE0 vs MSE1')

# maska MSE 0 a 1
dataMSE = pd.read_csv('mse.csv', delimiter=';')
# rozdelit do dvoch dataframov, v jednom budu len 0, v druhom len rovnake 1
maskaM = (dataMSE.value == 0)
dataMSE0 = dataMSE[maskaM]
dataMSE1 = dataMSE[~maskaM]
sns.histplot(data=dataMSE0, x='MSE', bins=20, color="pink", ax=axes[0])

sns.histplot(data=dataMSE1, x='MSE', bins=20, color="green", ax=axes[1])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('SSIM0 vs SSIM1')

# maska SSIM 0 a 1
dataSSIM = pd.read_csv('ssim.csv', delimiter=';')
maskaS = (dataSSIM.value == 0)
dataSSIM0 = dataSSIM[maskaS]
dataSSIM1 = dataSSIM[~maskaS]
sns.histplot(data=dataSSIM0, x='SSIM', bins=20, color="red", ax=axes[0])

sns.histplot(data=dataSSIM1, x='SSIM', bins=20, color="blue", ax=axes[1])

# plt.xlabel('ID')
# plt.ylabel('SSIM')
plt.show()
