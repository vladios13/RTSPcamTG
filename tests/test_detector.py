import cv2
from app import detector

# Тест разницы между кадрами
img1 = cv2.imread('1.jpg', 0)
img2 = cv2.imread('2.jpg', 0)
diff = detector.get_image_difference(img1, img2)
print('Image difference:', diff)
