import os
import json
import cv2
import numpy as np

# Ruta a la base de conocimiento
knowledge_base_dir = "../knowledge_base"

# Cargar todos los descriptores de la base de conocimiento
knowledge_data = []
for file_name in os.listdir(knowledge_base_dir):
    if file_name.startswith("descriptores_bbox_batch"):  # Solo cargar los descriptores de bboxes
        with open(os.path.join(knowledge_base_dir, file_name), "r") as file:
            knowledge_data.append(json.load(file))

# Configuración del FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Leer la nueva imagen que se quiere clasificar
new_image_path = "../test_images/test_1.JPG"  # Cambia esta ruta a la imagen que desees clasificar
new_image = cv2.imread(new_image_path)

if new_image is None:
    print(f"No se pudo leer la imagen: {new_image_path}")
else:
    # Convertir a escala de grises
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Detectar esquinas de Harris con menor umbral para detectar más esquinas
    gray_harris = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray_harris, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(harris_corners, None)

    # Normalizar la imagen de Harris para usarla con SIFT
    harris_image_uint8 = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Usar SIFT para obtener los descriptores de la nueva imagen
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(harris_image_uint8, None)

    if descriptors is not None:
        # Variable para almacenar la mejor coincidencia
        best_match = None
        best_score = float('inf')  # Puntuación más baja significa mejor coincidencia
        best_match_supercategory = "Unknown"
        best_match_name = "Unknown"
        min_match_count = 10  # Umbral mínimo para considerar una coincidencia como válida

        # Realizar matching con los descriptores de los bboxes en la base de conocimiento
        for data in knowledge_data:
            for image_name, image_info in data.items():
                # Obtener los descriptores de cada bbox
                for bbox_info in image_info.get("bboxes", []):
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
                        if m.distance < 0.8 * n.distance:  # Ratio de Lowe ajustado
                            good_matches.append(m)

                    # Calcular la puntuación basada en la suma de las distancias de las coincidencias
                    if len(good_matches) >= min_match_count:
                        score = sum([match.distance for match in good_matches]) / len(good_matches)

                        # Actualizar la mejor coincidencia si es necesario
                        if score < best_score:
                            best_score = score
                            best_match = image_name
                            best_match_supercategory = bbox_info.get("supercategory", "Unknown")
                            best_match_name = bbox_info.get("name", "Unknown")

        # Mostrar el resultado de la mejor coincidencia
        if best_match:
            print(f"En esta imagen hay '{best_match_name}' de la supercategoría '{best_match_supercategory}'.")
        else:
            print("No se encontraron buenas coincidencias.")
