import os
import cv2
import numpy as np
import json

# Directorios
input_dir = '../subset'
output_dir = '../knowledge_base'
output_file = os.path.join(output_dir, 'descriptors.json')
annotations_file = os.path.join(input_dir, 'subset_annotations.json')

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Leer el archivo de anotaciones principal
with open(annotations_file, 'r') as f:
    annotations_data = json.load(f)

# Crear un mapeo de image_id a categorías (supercategory y name)
image_metadata = {}
for img in annotations_data["images"]:
    image_metadata[img["file_name"]] = {"image_id": img["id"]}

for category in annotations_data["categories"]:
    for img in annotations_data["images"]:
        if img["id"] == category["id"]:
            image_metadata[img["file_name"]].update({
                "supercategory": category["supercategory"],
                "name": category["name"]
            })

# Inicializar diccionario para almacenar los descriptores
descriptors_dict = {}
contador = 0

# Recorrer todas las carpetas batch_*
for batch_dir in os.listdir(input_dir):
    batch_path = os.path.join(input_dir, batch_dir)
    if not os.path.isdir(batch_path) or not batch_dir.startswith('batch_'):
        continue

    # Lista de imágenes en el batch
    image_files = [f for f in os.listdir(batch_path) if f.endswith('.jpg') or f.endswith('.JPG')]

    for image_file in image_files:
        img_path = os.path.join(batch_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error al cargar la imagen {img_path}")
            continue

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar esquinas de Harris
        gray_harris = np.float32(gray)
        harris_corners = cv2.cornerHarris(gray_harris, blockSize=2, ksize=3, k=0.04)
        harris_corners = cv2.dilate(harris_corners, None)

        # Normalizar la imagen de Harris para usarla con SIFT
        harris_image_uint8 = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Usar SIFT para obtener los descriptores
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(harris_image_uint8, None)

        # Almacenar los descriptores
        if descriptors is not None:
            metadata = image_metadata.get(image_file, {})
            descriptors_dict[metadata.get("image_id", contador)] = {
                "file_name": img_path,
                "supercategory": metadata.get("supercategory", "unknown"),
                "name": metadata.get("name", "unknown"),
                "descriptors": descriptors.tolist()
            }

        contador += 1
        print(f'Procesada imagen {contador}: {image_file}')

# Guardar los descriptores en un archivo JSON
with open(output_file, 'w') as json_file:
    json.dump(descriptors_dict, json_file, indent=4, sort_keys=True)

cv2.destroyAllWindows()
