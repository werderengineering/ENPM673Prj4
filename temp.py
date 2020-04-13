import cv2

img1 = cv2.imread('./Car4/img/0001.jpg')
xl = 65
xu = 180
yl = 45
yu = 135
temp = img1[yl:yu,xl:xu]
temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
cv2.imwrite('./Car4/template.jpg',temp)