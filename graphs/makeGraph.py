import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# class Photo:
#     def __init__(self, id, first, second, value):
#         self.id = id
#         self.first = first
#         self.second = second
#         self.value = value

df = pd.read_csv('mse.csv', delimiter=';')
sns.histplot(data=df, x='ID', y='MSE', bins=20, color="pink")

plt.xlabel('ID')
plt.ylabel('MSE')
plt.show()
