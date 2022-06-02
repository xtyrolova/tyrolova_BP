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
plt.subplot(1, 2, 1)
plt.ylabel('Počet')
plt.xlabel('MSE hodnoty 0')
sns.histplot(data=dataMSE0, x='MSE', bins=20, color="#E6CFDC", ax=axes[0])
plt.subplot(1, 2, 2)
plt.ylabel('Počet')
plt.xlabel('MSE hodnoty 1')
sns.histplot(data=dataMSE1, x='MSE', bins=20, color="#7FBD8A", ax=axes[1])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('SSIM0 vs SSIM1')

# maska SSIM 0 a 1
dataSSIM = pd.read_csv('ssim.csv', delimiter=';')
maskaS = (dataSSIM.value == 0)
dataSSIM0 = dataSSIM[maskaS]
dataSSIM1 = dataSSIM[~maskaS]

plt.subplot(1, 2, 2)
plt.ylabel('Počet')
plt.xlabel('SSIM hodnoty 0')
sns.histplot(data=dataSSIM0, x='SSIM', bins=20, color="#F28D5E", ax=axes[0])
plt.subplot(1, 2, 1)
plt.ylabel('Počet')
plt.xlabel('SSIM hodnoty 1')
sns.histplot(data=dataSSIM1, x='SSIM', bins=20, color="skyblue", ax=axes[1])

# plt.xlabel('ID')
# plt.ylabel('SSIM')
plt.show()
