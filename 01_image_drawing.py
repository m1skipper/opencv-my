#
# Урок 01. Рисование поверх изображения, прямоугольники, круги, текст
#
# Документация: 
#   https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
# Статьи с примерами:
#   https://arboook.com/kompyuternoe-zrenie/operatsii-s-tsvetom-v-opencv3-i-python/
#   https://robotclass.ru/tutorials/opencv-video-text-drawings/
# !Замечание:
#   Обязательно в программе должен быть код cv2.waitKey, с какой-нибудь задержкой,
#   иначе не отобразится окно в RaspberryPi(в этот момент происходит обработка сообщений)
#

import cv2

# Загрузить изображение png/jpg (gif не поддерживается, потому что требует лицензию)
image = cv2.imread("./data/test.png")

# Отобразить изображение в окне с именем "Source image"
cv2.imshow("Source image", image)

# Создадим копию, чтобы не испортить исходное изображение
output = image.copy()

# shape - кортеж(массив) содержащий размер изображения,
# третья компонента количество байт на цвет.
print("Height", image.shape[0])
print("Width", image.shape[1])
print("Channels", image.shape[2])
(h, w, _) = image.shape

# Рисуем прямоугольник на изображении
# rectangle(изображение, координаты 1-го угла, координаты 2-го угла, цвет bgr/яркость, толщина линии)
# Левый верхний угол 0,0
# !Замечание: координаты в функциях opencv: (x,y)
# (при использовании image, как массива, наоборот вначале строки, потом столбцы)
(x1, y1) = (260, 80)
(x2, y2) = (410, 240)
cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 10)
cv2.imshow("Result", output)

# Рисуем линию
# line(изображение, координаты 1-й точки, координаты 2-й точки, цвет, толщина линии)
cv2.line(output, (0, 0), (w, h), (0, 0, 255), 5)
cv2.imshow("Result", output)

# Рисуем окружность
# circle(изображение, координаты центра, радиус, цвет, толщина линии)
cv2.circle(output, (w//2, h//2), 140, (0, 255, 255), 10)
cv2.imshow("Result", output)

# Пишем текст
# putText(изображение, текст, нижний-левый угол, шрифт, размер шрифта, цвет bgr, толщина линий)
# шрифт, один из:
# cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX
# можно комбинировать с FONT_ITALIC для получения наклонных букв
# Размер шрифта для увеличения в два раза — пишем 2, для уменьшения в 2 раза — 0.5
# !Возможны проблемы с русскими буквами, вместо них отображает ???
cv2.putText(output, "Opa gangam style!", (15, 150),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 0), 6) 
cv2.imshow("Result", output)

# Бесконечно ожидаем, нажатия любой клавиши
# waitKey(задержка в миллисекундах)
cv2.waitKey(0)

# запишем изображение на диск в формате png
cv2.imwrite("./data/test_drawing.png", output)

# Закрыть все окна 
# На RaspberryPi лучше закрывать все окна, могут остаться после завершения программы
cv2.destroyAllWindows()
