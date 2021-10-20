import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# class Photo:
#     def __init__(self, id, first, second, value):
#         self.id = id
#         self.first = first
#         self.second = second
#         self.value = value

fig, axes = plt.subplots(1, 2, figsize=(10,5))
fig.suptitle('MSE vs SSIM')

dataMSE = pd.read_csv('mse.csv', delimiter=';')
sns.histplot(data=dataMSE, x='ID', y='MSE', bins=20, color="pink", ax=axes[0])

# plt.xlabel('ID')
# plt.ylabel('MSE')
# plt.show()
dataSSIM = pd.read_csv('ssim.csv', delimiter=';')
sns.histplot(data=dataSSIM, x='ID', y='SSIM', bins=20, color="green", ax=axes[1])

# plt.xlabel('ID')
# plt.ylabel('SSIM')
plt.show()

