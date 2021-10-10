import cv2

img = cv2.imread("fei.jpeg", 1)
print(img.shape)

imgResize = cv2.resize(img, (300,200))
print(imgResize.shape)

cv2.imshow("Gray Image", imgResize)

cv2.waitKey(0)