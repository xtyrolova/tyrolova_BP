import cv2

image = cv2.imread("fei.jpeg", 1)

sizeX = image.shape[1]
sizeY = image.shape[0]

parts = 5
for i in range(0, parts):
    row=[]
    for j in range(0, parts):
        start_y = i * sizeY / parts
        end_y = i * sizeY / parts + sizeY / parts
        start_x = j * sizeX / parts
        end_x = j * sizeX / parts + sizeX / parts
        #łroi = grayImage[int(start_y):int(end_y), int(start_x):int(end_x)]