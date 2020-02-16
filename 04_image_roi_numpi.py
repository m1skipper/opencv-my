#
# Урок 04. Регион интереса(ROI). Работа с массивами numpy, фокусы
#
# https://docs.scipy.org/doc/numpy-1.13.0/index.html
#

# Математическая библиотека для работы с многомерными массивами и матричных вычислений
import numpy as np

import cv2

image = cv2.imread("./data/test.png")
cv2.imshow("Source image", image)

# Изображение это 2-х, или 3-х мерный массив numpy.ndarray
# Первое измерение строки, второе столбцы, а третье массив цветов [b,g,r]/[h,s,v], либо число для черно белых изображений.
print('Класс изображения:', type(image))
print('Размерность элемента данных:', image.dtype)

# shape - размерность матрицы - кортеж(tuple) из 3-х чисел высота, ширина, длина массива цвета в байтах
print('Трехмерная цветная матрица:', image.shape)
print("Height", image.shape[0])
print("Width", image.shape[1])
print("Channels", image.shape[2])
(h, w, _) = image.shape

print('Первый пиксель цветного изображения(массив байт):', image[0][0]) 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Первый пиксель изображения в градациях серого(число uint8):', gray_image[0][0]) 

# На самом деле gray_image двумерный массив
print("Двумерная черно-белая матрица:",gray_image.shape)

# ! При работе с изображениями как с массивами первая координата y(номер строки), вторая x(номер столбца)
print('Последний пиксель цветного изображения:', image[h-1][w-1]) 

# Это одно и то же изображение(копируется только ссылка), правим sameimage изменяется image
sameimage = image

# А это уже копия изображения, если править newimage, image останется не тронутым
imagecopy = image.copy()

# ROI - region of interest, регион интереса, часть изображения.
# Прямоугольный подмассив в изображении(срез массива numpy).
# ! В отличие от обычного массива, срез numpy.ndarray не является копией, 
# а ссылется на исходное изображение, т.е. если править данные среза, то правятся данные исходного изображения
cropped = image[70:460, 150:450]
cv2.imshow("Cropped image", cropped)

# Обрежем слева и справа по 1/3
cropped = image[0:h, w//3:w-w//3]
cv2.imshow("Middle", cropped)

# Срез можно делать по любому массиву
(h, w) = image.shape[0:2]
# То же самое
print('(h,w):', image.shape[:2])

print('Синий цвет первого пикселя(число uint8):', image[0][0][0]) 
print('Зеленый цвет первого пикселя(число uint8):', image[0][0][1]) 
print('Красный цвет первого пикселя(число uint8):', image[0][0][2]) 

# Как выглядят отдельные комоненты(двумерные массивы содержащие один цвет)
# Помните, что первая компонента b, потом g, потом r.
b_image = image[:,:,0]
g_image = image[:,:,1]
r_image = image[:,:,2]

# Собрать картинку назад, в обратном порядке, где красный это синий
imagerev = np.zeros((h, w, 3), np.uint8)
imagerev[:,:,2] = b_image
imagerev[:,:,1] = g_image
imagerev[:,:,0] = r_image
cv2.imshow("Image rev", imagerev)

# Создать пустое(черное) изображение, это создать пустую матрицу numpy.ndarray.
# zeros((высота, ширина, кол-во байт цвета), тип данных элемента)
image3 = np.zeros((h, w, 3), np.uint8)

# Можно не только делать срезы, но и копировать в срезы новые данные
# Соберем картинку из 3-х разных компонент
image3[:,0:w//3,2] = r_image[:,0:w//3]
image3[:,w//3:w*2//3,1] = g_image[:,w//3:w*2//3]
image3[:,w*2//3:w,0] = b_image[:,w*2//3:w]
# Магия? -Не думаю.
cv2.imshow("3 colors", image3)

# Соединить несколько массивов(изображений) в один
# np.concatenate((изображение1, изображение2,...), axis)
# axis(ось) = 1 по горизонтали, 0 по вертикали, 
# ! Картинки должны быть одинакового размера по второй плоскости и иметь одинаковую размерность цвета
left_half_image = image[:,0:w//2]
right_half_image = imagerev[:,w//2:]
image2 = np.concatenate((left_half_image, right_half_image), axis=1)
cv2.imshow("2 images", image2)

# Ещё вариант собрать картинку из разных цветовых плоскостей
# Из двумерного массива сделаем 3-х мерный, но последняя компонента цвета имеет длину 1
# reshape - изменить форму массива, параметр содержащий длинну новых измерений, общее количество элементов
# исходного массива должно быть такое же как и у массива новой формы. Можно менять h,w местами у массива,
# можно, из 2д делать 3д, или 1д и наоборот. Новый массив не копия, а представление, т.е. те же данные,
# только с другим способом доступа.
r_image_1 = r_image.reshape((h,w,1))
g_image_1 = g_image.reshape((h,w,1))
b_image_1 = b_image.reshape((h,w,1))
# Теперь мы можем склеить картинки по третей размерности axis=2
imagerev2 = np.concatenate((b_image_1, r_image_1, g_image_1), axis=2)
cv2.imshow("Image rev2",imagerev2)

# Фильтры
image_black_around = np.zeros((h, w, 3), np.uint8)
image_black_around[150:360,480:720] = image[150:360,480:720]
cv2.imshow("Image black around", image_black_around)

# К изображению, или срезу можно применить условие, результатом будет матрица того же размера
# содержащая true/false, где условие применяется к каждому элементу
filterimage = image_black_around[:,:,0]>100

# Чтобы можно было отобразить, как картинку нужно преобразовать true/false в байт,
# например так(true*255 = 255, false*255=0)
# после уножения тип элементов будет int, а нам нужен uint8, поэтому сменим тип в матрице через astype
blue_image_thresholded = (filterimage*255).astype(np.uint8)
cv2.imshow("Blue image thresholded", blue_image_thresholded)

# Создадим полуинтервал [firstIndex, lastIndex) из индексов, аналогично range, только результат будет numpy.ndarray
# arrange(firstIndex, lastIndex)
x_indexes = np.arange(0,blue_image_thresholded.shape[1])
y_indexes = np.arange(0,blue_image_thresholded.shape[0])

# Создаем матрицу из повторяющихся строк
x_indexes_matrix = x_indexes.reshape(1, x_indexes.shape[0])
x_indexes_matrix = np.repeat(x_indexes_matrix, [y_indexes.shape[0]], axis=0)

# Создаем матрицу из повторяющихся столбцов
y_indexes_matrix = y_indexes.reshape(y_indexes.shape[0],1)
y_indexes_matrix = np.repeat(y_indexes_matrix, [x_indexes.shape[0]], axis=1)

# Функции над массивами
# Фильтр можно применить к матрице, тогда получим элементы, которые соответствуют истинным условиям
# Результат в массиве останутся только те элементы для которых фильтр true
x_not_null_indexes = x_indexes_matrix[filterimage]
y_not_null_indexes = y_indexes_matrix[filterimage]

# К массиву можно применить функцию, например, sum, any, all, min, max
# Сумма x индексов отличных от 0
xsum = x_not_null_indexes.sum() 
# Сумма x индексов отличных от 0
ysum = y_not_null_indexes.sum() 
# Всего не нулевых точек
point_count = filterimage.sum()

# Как найти центр центральную точку изображения:
# x_center = сумму всех x коорданат ненулевых точек, деленная на количество ненулевых точек,
# y_center = сумму всех y коорданат ненулевых точек, деленная на количество ненулевых точек.
x_center = xsum/point_count
y_center = ysum/point_count

print("(x_center, y_center):", (x_center, y_center))
center = np.int0((x_center, y_center))
cv2.circle(blue_image_thresholded, (int(x_center), int(y_center)), 100, 255, 5)
cv2.imshow("Blue image thresholded circle", blue_image_thresholded)

# Ещё есть классная вещь, функции вдоль оси. Возвращают матрицы вычисленных значений
# Например: any(axis=2), all(axis=0), sum(axis=1), 
# np.apply_along_axis(my_axis_func, 2, image) - медленная потому что for не использует оптимизацию
# Пример склеивания картинок с использованием черного цвета как прозрачного
# Создадим просто синюю картинку
blue_pic = np.full((h,w,3),(255,0,0), np.uint8)
# Создадим маску, где любой не черный цвет это 255, любой черный 0.
mask = (image_black_around.any(axis=2)).astype(np.uint8)*255
mask_inv = cv2.bitwise_not(mask)
# А дальше обычная побитная склейка по маске
union_image = cv2.bitwise_and(image_black_around,image_black_around,mask=mask)
union_image2 = cv2.bitwise_and(blue_pic,blue_pic,mask=mask_inv)
union_image = cv2.bitwise_or(union_image,union_image2)
cv2.imshow("Union image", union_image)

# Ещё один вариант накладывание изображений с прозрачным черным(0,0,0), чисто numpy
mask_filter = image_black_around.any(axis=2)
union_image2 = blue_pic.copy()
union_image2[mask_filter] = image_black_around[mask_filter]
cv2.imshow("Union image2", union_image2)

# В общем, учите мат часть!))) Как работать с массивами в python и с матрицами в numpy
# Очень полезно для быстрой обработки изображений.
# Циклы for/while крайне медленные в python, функции numpy и opencv быстрые
# (используют распараллеливание и возможности процессора: sse,...).

cv2.waitKey(0)
cv2.destroyAllWindows()
