#
# Урок 02. Трансформация. Изменение размера, поворот
#
# https://arboook.com/kompyuternoe-zrenie/osnovnye-operatsii-s-izobrazheniyami-v-opencv-3-python/
#

import cv2
import numpy as np

# Загрузить изображение
image = cv2.imread("./data/test.png")
cv2.imshow("Source image", image)
cv2.waitKey(0)

print('Размер изображения', image.shape)
(h, w, _) = image.shape

# Изменим размер изображения
# resize(изображение, (ширина, высота), fx=растяжение по горизонтали, fy=растяжение по вертикали, interpolation=метод)
# метод: cv2.INTER_AREA, cv.INTER_LINEAR(по умолчанию) и др. 
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/ 

# Пропорционально изменим стороны
new_width = 400
new_height = int(h*(new_width/w))
print("(new_width,new_width):",(new_width, new_height))
resized_image = cv2.resize(image, (new_width, new_height))
cv2.imshow("Resize image", resized_image)
cv2.waitKey(0)

# Увеличить в полтора раза
resized_image2 = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize image2", resized_image2)
cv2.waitKey(0)

# Отразить изображение
# flip(изображение, плоскость)
# плоскость: 0 – по вертикали, 1 – по горизонтали, (-1) – по вертикали и по горизонтали.
flip_image = cv2.flip(image,1) 
cv2.imshow("Flip image", flip_image)
cv2.waitKey(0)

# Повернем изображение на 180
# Поворот осуществляется с помощью матриц преобразования
# getRotationMatrix2D(точка центр поворота, угол в градусах, масштаб)
# Повернем изображение на 180 градусов, не меняя размер
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, 180, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated 180 image", rotated)
cv2.waitKey(0)

# Сдвиг изображения с помощью матрицы трансляции(сдвига)
dx = 100
dy = 10
T = np.float32([[1, 0, dx], [0, 1, dy]]) 
translated_image = cv2.warpAffine(image, T, (w+dx, h+dy))
cv2.imshow("Translated image", translated_image)
cv2.waitKey(0)

# Поворот на 90 со смещением
# При повороте на другие углы, чтобы всё изображение вошло, нужно изменить размер изображения
# и передвинуть картинку.
M = cv2.getRotationMatrix2D((0,0), 90, 1)
# добаляем в преобразование сдвиг
dx = 0
dy = w
M[0][2] += dx
M[1][2] += dy
rotated = cv2.warpAffine(image, M, (h, w))
cv2.imshow("Rotated 90 image", rotated)
cv2.waitKey(0)

# вырежем участок изображения используя срезы numpy
cropped = image[70:460, 150:450]
cv2.imshow("Cropped image", cropped)
cv2.waitKey(0)

# 4-х точечная перспективная трансформация
# как пользоваться для проецирования изображения и восстановления
# см. mycv.imageFrom4PointContour, mycv.imageTo4PointsContour
rect = np.array([[0,0], [0,h], [w,h], [w,0]], dtype = "float32")
dst = np.array([[0,0], [w/3,h], [2/3*w,h*2.0/3], [w*2/3,h/3]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect,dst)
resultWidth = w
resultHeight = h
warped = cv2.warpPerspective(image, M, (resultWidth, resultHeight))
cv2.imshow("Warped image", warped)
cv2.waitKey(0)

cv2.destroyAllWindows()
