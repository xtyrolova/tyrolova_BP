import cv2
import matplotlib.pyplot as plt

image = cv2.imread("fei.jpeg", 1)

grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)

sizeX = image.shape[1]
sizeY = image.shape[0]

parts = 5
f, ax = plt.subplots(parts, parts, figsize=(10,5), num="Parts")
for i in range(0, parts):
    row=[]
    for j in range(0, parts):
        start_y = i * sizeY / parts
        end_y = i * sizeY / parts + sizeY / parts
        start_x = j * sizeX / parts
        end_x = j * sizeX / parts + sizeX / parts
        roi = grayImage[int(start_y):int(end_y), int(start_x):int(end_x)]
        ax[i][j].imshow(roi)
        ax[i][j].axis('off') #hide the axis
        f.tight_layout()
plt.show()

