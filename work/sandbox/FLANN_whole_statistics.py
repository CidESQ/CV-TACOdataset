import os
import json
import cv2
import numpy as np
from pathlib import Path

# Rutas
subset_dir = "../subset"
knowledge_base_dir = "../knowledge_base"
annotations_file = os.path.join(subset_dir, "subset_annotations.json")

# Cargar el archivo de anotaciones JSON
with open(annotations_file, "r") as file:
    annotations = json.load(file)

# Obtener el mapeo de categorías para un acceso más sencillo
categories = {category['id']: category for category in annotations['categories']}

# Configuración del FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Cargar todos los descriptores de la base de conocimiento
knowledge_data = []
for file_name in os.listdir(knowledge_base_dir):
    if file_name.startswith("descriptores_bbox_batch"):  # Solo cargar los descriptores de bboxes
        with open(os.path.join(knowledge_base_dir, file_name), "r") as file:
            knowledge_data.append(json.load(file))

# Inicializar contadores de aciertos y fallos
total_aciertos = 0
total_fallos = 0
total_imagenes = 0

# Recorrer todos los batches del 1 al 15
for batch_num in range(1, 16):
    batch_dir = os.path.join(subset_dir, f"batch_{batch_num}")

    # Procesar cada imagen en el directorio batch
    for img_info in annotations['images']:
        if not img_info['file_name'].startswith(f"batch_{batch_num}/"):
            continue

        image_path = os.path.join(subset_dir, img_info['file_name'])
        image_id = img_info['id']

        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            continue

        # Obtener las anotaciones para la imagen
        annotations_for_image = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        if not annotations_for_image:
            continue
        
        total_imagenes += 1
        print(f"Procesando imagen: {img_info['file_name']} (ID: {image_id})")

        # Paso 2: Usar SIFT y Harris para Extraer los Descriptores
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_harris = np.float32(gray)
        harris_corners = cv2.cornerHarris(gray_harris, blockSize=2, ksize=3, k=0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        harris_image_uint8 = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(harris_image_uint8, None)

        if descriptors is None:
            print("No se pudieron extraer descriptores de la imagen.")
            continue

        # Paso 3: Realizar el Matching con la Base de Conocimiento
        best_match_supercategory = "Unknown"
        best_match_name = "Unknown"
        best_score = float('inf')
        min_match_count = 10

        for data in knowledge_data:
            for knowledge_img_name, knowledge_img_info in data.items():
                # Obtener los descriptores de cada bbox
                for bbox_info in knowledge_img_info.get("bboxes", []):
                    if "descriptors" not in bbox_info:
                        continue

                    knowledge_descriptors = np.array(bbox_info["descriptors"], dtype=np.float32)

                    # Comprobar si hay suficientes descriptores para hacer knnMatch con k=2
                    if len(knowledge_descriptors) < 2:
                        continue

                    # Hacer matching con FLANN
                    matches = flann.knnMatch(descriptors, knowledge_descriptors, k=min(2, len(knowledge_descriptors)))

                    # Aplicar la prueba de ratio de Lowe para filtrar buenas coincidencias
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.8 * n.distance:
                            good_matches.append(m)

                    # Calcular la puntuación basada en la suma de las distancias de las coincidencias
                    if len(good_matches) >= min_match_count:
                        score = sum([match.distance for match in good_matches]) / len(good_matches)

                        # Actualizar la mejor coincidencia si es necesario
                        if score < best_score:
                            best_score = score
                            best_match_supercategory = bbox_info.get("supercategory", "Unknown")
                            best_match_name = bbox_info.get("name", "Unknown")

        # Paso 4: Comparar con las Anotaciones y Registrar Aciertos/Fallos
        for ann in annotations_for_image:
            expected_supercategory = categories[ann['category_id']]['supercategory']
            expected_name = categories[ann['category_id']]['name']

            if best_match_supercategory == expected_supercategory and best_match_name == expected_name:
                total_aciertos += 1
                print("SI hubo coincidencia")
            else:
                total_fallos += 1
                print("NO hubo coincidencia")

        print(f"Resultado para la imagen {img_info['file_name']}: Mejor coincidencia - Supercategory: {best_match_supercategory}, Name: {best_match_name}")

# Mostrar estadísticas de rendimiento
total_comparaciones = total_aciertos + total_fallos
print(f"Total de imágenes procesadas: {total_imagenes}")
print(f"Aciertos: {total_aciertos}")
print(f"Fallos: {total_fallos}")
print(f"Total de comparaciones: {total_comparaciones}")
if total_comparaciones > 0:
    print(f"Precisión: {(total_aciertos / total_comparaciones) * 100:.2f}%")
else:
    print("No se realizaron comparaciones.")
