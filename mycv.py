#
# Вспомогательные функции для работы с opencv
#

# Математическая библиотека для работы с многомерными массивами и матричных вычислений
import numpy as np
import cv2

# Увеличить уменьшить изображение
def scale(image, fx, fy = None):
	if fy is None:
		fy = fx
	return cv2.resize(image, None, fx=fx, fy=fy, interpolation = cv2.INTER_AREA)

# Изменим пропорционально по ширине
def resizeWidth(image, width):
	(h, w, _) = image.shape
	new_width = width
	new_height = int(h*(new_width/w))
	resized_image = cv2.resize(image, (new_width, new_height))
	return resized_image

# Изменим пропорционально по ширине
def resizeHeight(image, height):
	(h, w, _) = image.shape
	new_height = height
	new_width = int(w*(new_height/h))
	resized_image = cv2.resize(image, (new_width, new_height))
	return resized_image

# Поворот на угол в градусах(размер изображения не меняется)
def rotate(image, angle):
	(h, w, _) = image.shape
	M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
	return cv2.warpAffine(image, M, (w, h))

# Поворот на 90 несколько раз(форма изображения меняется)
def rotate90(image, count = 1):	
	(h, w, _) = image.shape
	newdim = (w,h)
	M = cv2.getRotationMatrix2D((0,0), count*90, 1)
	if count==0:
		return image
	elif count==1:
		M[0][2] += 0
		M[1][2] += w
		newdim = (h,w)
	elif count==2:
		M[0][2] += w
		M[1][2] += h
		newdim = (w,h)
	elif count==3:
		M[0][2] += h
		M[1][2] += 0
		newdim = (h,w)
	rotated = cv2.warpAffine(image, M, newdim)
	return rotated

# Добавить на первую картинку, вторую.
# Черные точки второй картинки являются прозрачными
def imageOnImage(image1, image2):
	mask_filter = image2.any(axis=2)
	newimage = image1.copy()
	newimage[mask_filter] = image2[mask_filter]
	return newimage

# Возвращает среднюю точку изображения (x, у, и площадь)
# На вход черно белое изображение
def moments(threshImage):
	moments = cv2.moments(threshImage, 1)
	dM01 = moments['m01']
	dM10 = moments['m10']
	dArea = moments['m00']
	x = dM10 / dArea
	y = dM01 / dArea
	return (int(x),int(y),int(dArea))

# Сортировка контуров по площади по убыванию с ограничением до count контуров
# contours - контуры возвращенные findContours
# count - сколько контуров оставить в массиве
def sortContours(contours, count = 0):
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    if  count != 0:
    	return contours[:count]
    return contours

# Найти приблизительный наибольший 4-х точечный контур
def find4PointsContour(contours):

  # Рассматривать будет только 10 самых больших
  cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

  our_cnt = None

  for c in cnts:
      # approximate the contour
      # These methods are used to approximate the polygonal curves of a contour. 
      # In order to approximate a contour, you need to supply your level of approximation precision. 
      # In this case, we use 2% of the perimeter of the contour. The precision is an important value to consider. 
      # If you intend on applying this code to your own projects, you’ll likely have to play around with the precision value.
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4:
          our_cnt = approx
          break

  return our_cnt

# Спроецировать изображение img на изображение bg в 4-х точечный контур cnt
def imageTo4PointsContour(img, bg, cnt):

    # we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # summing the (x, y) coordinates together by specifying axis=1
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Notice how our points are now stored in an imposed order: 
    # top-left, top-right, bottom-right, and bottom-left. 
    # Keeping a consistent order is important when we apply our perspective transformation

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # если на помещаемой картинке есть совсем черные точки, сделаем их не совсем черными
    img_contour_sized = cv2.resize(img, (maxWidth, maxHeight), cv2.INTER_AREA)
    mask = img_contour_sized.any(axis=2)
    mask = np.invert(mask)
    img_contour_sized[mask] = [1,1,1]

    # calculate the perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(dst, rect)
    warp = cv2.warpPerspective(img_contour_sized, M, (bg.shape[1], bg.shape[0]))

    # вырежем bg внутри контура и добавим туда преобразованную картинку
    mask = (warp.any(axis=2)).astype(np.uint8)*255
    mask = cv2.bitwise_not(mask)
    bg_holed = cv2.bitwise_and(bg,bg,mask=mask)
    img_result = cv2.bitwise_or(bg_holed, warp)

    return img_result

def imageFrom4PointContour(orig, cnt):
    # we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # summing the (x, y) coordinates together by specifying axis=1
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Notice how our points are now stored in an imposed order: 
    # top-left, top-right, bottom-right, and bottom-left. 
    # Keeping a consistent order is important when we apply our perspective transformation

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect,dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    return warp



'''
# возвращает x координату линии(для движения робота по линии)
def detectLine(image):
	return None

image = cv2.imread("./data/test.png")

im = scale(image, 0.5)
cv2.imshow("scale 0.5",im)
cv2.waitKey(0)

cv2.imshow("w",rotate90(resizeWidth(image, 300)))
cv2.waitKey(0)
cv2.imshow("h",resizeHeight(image, 300))
cv2.waitKey(0)

im2 = cv2.imread("./data/test.png")
im2=rotate(im2,90)
cv2.imshow("22",im2)
cv2.waitKey(0)
ret = addImage(image, im2)
cv2.imshow("2",ret)
cv2.waitKey(0)
'''