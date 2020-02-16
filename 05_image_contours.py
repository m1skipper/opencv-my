#
# Урок 05. Выделение контуров.
# https://robotclass.ru/tutorials/opencv-detect-rectangle-angle/
# https://answers.opencv.org/question/67150/using-minarearect-with-contour-in-python/
#

import sys
import numpy as np
import cv2
import math

img = cv2.imread('./data/donut.jpg')
hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
thresh = cv2.inRange( hsv, (0, 54, 5), (187, 255, 253) ) # применяем цветовой фильтр

# Находим контуры на рисунку
# cv2.RETR_TREE - все контуры сгруппированные в иерарзию
# cv2.RETR_LIST - просто все контуры в списке
# cv2.RETR_EXTERNAL - внешние контуры
# С изображения нужно снять копию, если будет использоваться дальше, потому что портит данные
contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ret, contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) для старой версии библиотеки
# contours = cv2.findContours( thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-1] - универсальный метод для всех версий

# сортировка контуров по площади по убыванию с ограничением до 4-х контуров
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:4]
    
# перебираем все найденные контуры в цикле
for cnt in contours:
  rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
  # rect - повернутый п/у первая часть координата угла, вторая ширина высота, третья угол поворота
  print('([x,y], [width, height], angle): ', rect)
  area = int(rect[1][0]*rect[1][1]) # вычисление площади

  if area > 500: # отсекем лишние паразитные контуры по площади
    box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника 
    box = np.int0(box) # округление координат до целого

    cv2.drawContours(img,[box],0,(255,0,0),2)

  if len(cnt)>4:
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(img,ellipse,(0,0,255),2)

cv2.imshow('contours', img)
cv2.waitKey()
cv2.destroyAllWindows()

# ещё пример
color_yellow = (0,255,255)
color_red = (0,0,255)
img = cv2.imread('./data/pis.jpg')

hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
thresh = cv2.inRange( hsv, (0, 54, 5), (187, 255, 253) ) # применяем цветовой фильтр
contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# перебираем все найденные контуры в цикле
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
    box = np.int0(box) # округление координат

    center = (int(rect[0][0]),int(rect[0][1]))
    area = int(rect[1][0]*rect[1][1])
    angle = int(rect[2])

    if area > 500:
        cv2.drawContours(img,[box],0,(255,0,0),2) # рисуем прямоугольник
        cv2.circle(img, center, 5, color_yellow, 2) # рисуем маленький кружок в центре прямоугольника
        # выводим в кадр величину угла наклона
        cv2.putText(img, str(int(angle)), (center[0]+20, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color_red, 2)

cv2.imshow('contours', img)
cv2.waitKey()
cv2.destroyAllWindows()

# найдем не п/у 4-х точечные контуры
img = cv2.imread('./data/1000.jpg')
hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
thresh = cv2.inRange( hsv, (0,0,0),(255,70,255) ) # применяем цветовой фильтр
cnts = cv2.findContours( thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

# Нарисовать все контуры
cv2.drawContours(img, cnts, -1, (0, 255, 0), 3) 

print(len(cnts))

# ищем среди 10 самых больших по площади контуров
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
our_cnt = None
for c in cnts:
    # Найти приблизительный контур
    # Для этого необходимо задать точность приближения
    # В примере используем 2% периметра контура
    peri = cv2.arcLength(c, True) # посчитаем длину(периметр) контура
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # Если приблизительный контур имеет 4 точки, то это то что нам нужно
    # Будем считать что это проекция прямоугольника.
    if len(approx) == 4:
        our_cnt = approx
        break

if not our_cnt is None:
    cv2.drawContours(img,[our_cnt],0,(255,0,0),2) # рисуем прямоугольник

cv2.imshow('tr', thresh)
cv2.imshow('cnt', img)
cv2.waitKey()
cv2.destroyAllWindows()
