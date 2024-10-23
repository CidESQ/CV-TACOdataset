# Cid Emmanuel Esquivel Gonzalez

import cv2
import numpy as np
import os

# Deteccion de esquinas de Harris

# image_dir = '../../'
image_dir = '../subset/batch_1'

# Lista todas las imagenes en el directorio
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error al cargar la imagen {img_path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    #detecta las esquinas de Harris
    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    # mostrar la imagen con las esquinas detectadas
    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
