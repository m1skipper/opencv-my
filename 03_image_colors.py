#
# Урок 03. Работаем с цветом
# Преобразование цветовых пространств, разложение на компоненты, фильтры, выделение цветов
#
# https://tproger.ru/translations/opencv-python-guide/
# https://arboook.com/kompyuternoe-zrenie/operatsii-s-tsvetom-v-opencv3-i-python/
# https://robotclass.ru/tutorials/opencv-video-rgb-hsv-gray/
#

import cv2

# Загрузить изображение
image = cv2.imread("./data/homepage-bild-Ho.jpg")
cv2.imshow("Source image", image)
cv2.waitKey(0)

# Функция преобразования цветов в другое цветовое пространство:
# cvtColor(изображение, цветовое пространство)
# цветовое пространство(примеры): cv2.BGR2HSV, cv2.BGR2BGRAY, cv2.BGR2BGRA, cv2.BGR2RGB
# (конвертировать можно из любого в любое)

# Переведем в градации серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", gray_image)
cv2.waitKey(0)

# Для прикола как выглядит rgb в hsv(чтобы определять, что картинка не в том формате)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV in RGB image", hsv_image)
cv2.waitKey(0)

# Как выглядят отдельные комоненты
cv2.imshow('Hue',hsv_image[:,:,0])
cv2.waitKey(0)
cv2.imshow('Saturation',hsv_image[:,:,1])
cv2.waitKey(0)
cv2.imshow('Value',hsv_image[:,:,2])
cv2.waitKey(0)

# По умолчанию отображается BGR, для прикола поменяем Blue и Red местами
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("BGR to RGB for fun", rgb)
cv2.waitKey(0)

# Пороговый фильтр(threshold-порог) 
# threshold(изображение, порог, максимальное значение(новое), тип)
# порог - значение от 0 до 255 по которому разделяем,
# максимальное значение - то что будет поставлено для значений больше порога при параметре тип=BINARY,BINARY_INV
# тип - cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV 
# Описания типов по адресу https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

# Преобразование в двухцветное используя пороговый фильтр
# Вначале из цветного делаем серое, а потом из серого черно-белое
retval, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Black and white image", bw_image)
cv2.waitKey(0)

# Адаптивный порог. Автоматически, алгоритмически подбирается пороговое значение
# adaptiveThreshold(изображение, новое значение, метод подбора, тип порога, размер блока, константа)
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
threshA = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("Adaptive thresholded", threshA)
cv2.waitKey(0)

# Выделение цветом
# inRange(изображение, минимальный цвет, максимальный цвет)
# Получим маску, где белые точки соотвутствуют красным точкам
# (Красный цвет это когда красного много, а остальных цветов мало).
minRed = (0,0,200)
maxRed = (127,127,255)
mask_red = cv2.inRange(image, minRed, maxRed)  
cv2.imshow("Mask red", mask_red)
cv2.waitKey(0)

# Для выделения цветов удобнее использовать цветовое пространство HSV 
# hue(насыщенность-цвет) любое, saturation - не белый, и value - небольшое значит черный
# ! Настраивая фильтр, вначале ставим полный (0,0,0)-(255,255,255), а потом уменьшаем его, чтобы только наш 
# цвет попал.
black_hsv_min = (0,0,0)
black_hsv_max = (255,127,50)
mask_black = cv2.inRange(hsv_image, black_hsv_min, black_hsv_max)
cv2.imshow("Mask black", mask_black)
cv2.waitKey(0)

# Для фильтра красного в hsv придётся комбинировать 2 фильтра, потому что красный диапазон hue примерно 200-55 
# переходит через 0.
redMin1 = (230,10,100)
redMax1 = (255,235,255)
redMin2 = (0,10,100)
redMax2 = (25,235,255)
mask_red1 = cv2.inRange(hsv_image, redMin1, redMax1)
mask_red2 = cv2.inRange(hsv_image, redMin2, redMax2)
mask_red = cv2.bitwise_or(mask_red1,mask_red2)
cv2.imshow("Mask red hsv", mask_red)
cv2.waitKey(0)

# Размытие изображения
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# GaussianBlur(изображение, размер ядра, отклонение)
blurred = cv2.GaussianBlur(image, (51, 51), 0)
cv2.imshow('GaussianBlur', blurred)
cv2.waitKey(0)

# Медианный фильтр убирает salt and peper noise
# (шум соли и перца, чередование белых и черных частиц встречается на видеоизображениях), 
# берет среднее значение в радиусе
# medianBlur(изображение, ядро)
image = cv2.medianBlur(image,5)
cv2.imshow('MedianBlur', blurred)
cv2.waitKey(0)

# Размытие используется в том числе, чтобы сгладить границы, для лучшего последующего выделения контуров
# Применяется к серому изображению
blur = cv2.GaussianBlur(gray_image,(5,5),0)
# Otsu параметр в threshold методе позволяет не передавать в него значение порога, наоборот
# он будет вычислен автоматически и вернется в первом параметре
thres_val, threshO = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Thresh value", thres_val)
cv2.imshow('Otsu thresholded', threshO)
cv2.waitKey(0)
