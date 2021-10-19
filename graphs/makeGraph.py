import csv

# class Photo:
#     def __init__(self, id, first, second, value):
#         self.id = id
#         self.first = first
#         self.second = second
#         self.value = value

data = open('dtb.csv', 'r')
file = csv.DictReader(data, delimiter=';')

ids = []
images1 = []
images2 = []
values = []

for col in file:
    col.keys()
    ids.append(col['ID'])
    images1.append(col['path1'])
    images2.append(col['path2'])
    values.append(col['value'])

# print('ids:', ids)
# print('img1:', images1)
# print('img2:', images2)
# print('val:', values)