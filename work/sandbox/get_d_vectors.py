# Cid Emmanuel Esquivel Gonzalez

import cv2
import numpy as np
import os
import json

# Deteccion de esquinas de Harris

image_dir = '../../'
# image_dir = '../subset/batch_1'
output_file = '../subset/descriptors.json'

# Lista todas las imagenes en el directorio
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

#Inicializar diccionario para almacenar los descriptores
descriptors_dict = {}

for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error al cargar la imagen {img_path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_harris = np.float32(gray)
    
    #detecta las esquinas de Harris
    harris_corners = cv2.cornerHarris(gray_harris, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(harris_corners, None)

    # Marcar las esquinas en la imagen original
    img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    # Implementacion aproximada de MOPS usando SIFT para obtener los descriptores
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # Almacenar los descriptores en el diccionario
    if descriptors is not None:
        image_id = image_file #Aquí utilizar un mapeo con el archivo JSON principal si se utiliza
        descriptors_dict[image_id] = descriptors.tolist() # Convertir los descriptores a una lista
    
    # mostrar la imagen con las esquinas detectadas
    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)

# Guardar los descriptores en un archivo JSON
with open(output_file, 'w') as json_file:
    json.dump(descriptors_dict, json_file, indent=4)

cv2.destroyAllWindows()
