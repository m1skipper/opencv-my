#
# Урок 06. Убираем мелкие дефекты после применения выделения цветом
#

#TODO:

#
# https://www.youtube.com/watch?v=DYGGDz-4q4o
# Erode и Dilate функции для убирания ряби. Первая убирает отдельно стоящие белые точки. 
# Вторая укрупняет белые области. (Применяется к маске)

# основное изображение делаем blur - размываем, чтобы избавиться от цветовых шумов.
# blur делаем для hsv

#nакже можно не делать отдельно эрозию и дилатацию, а применить функцию morphologyEx с параметром MORPH_CLOSE 

# ещё варианты
# 

#Удалить мелкие объекты http://qaru.site/questions/239924/remove-spurious-small-islands-of-noise-in-an-image-python-opencv
#https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
#http://qaru.site/questions/15913717/opencv-how-to-correctly-apply-morphologyex-operation

import matplotlib.pyplot as plt
from skimage import morphology
import numpy as np
import skimage

# read the image, grayscale it, binarize it, then remove small pixel clusters
im = plt.imread('spots.png')
grayscale = skimage.color.rgb2gray(im)
binarized = np.where(grayscale>0.1, 1, 0)
processed = morphology.remove_small_objects(binarized.astype(bool), min_size=2, connectivity=2).astype(int)

# black out pixels
mask_x, mask_y = np.where(processed == 0)
im[mask_x, mask_y, :3] = 0

# plot the result
plt.figure(figsize=(10,10))
plt.imshow(im)
