import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
new_image_path = "../test_images/test_5.JPG"  # Cambia esta ruta a la imagen que desees clasificar
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
        best_match_keypoints = []
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
                            best_match_keypoints = good_matches

        # Mostrar el resultado de la mejor coincidencia
        if best_match:
            # Obtener la ruta correcta para la imagen encontrada en la base de conocimiento
            # Verificar si el nombre de la imagen ya contiene la información del batch
            if "batch_" in best_match:
                image_path = os.path.join("../subset", best_match)
            else:
                # Caso donde solo se tiene el nombre del archivo, sin la carpeta del batch
                for batch_num in range(1, 16):  # Intentar encontrar en cada batch
                    possible_path = os.path.join("../subset", f"batch_{batch_num}", best_match)
                    if os.path.exists(possible_path):
                        image_path = possible_path
                        break
                else:
                    print(f"No se encontró la imagen en ninguno de los batches: {best_match}")
                    image_path = None

            if image_path and os.path.exists(image_path):
                matched_image = cv2.imread(image_path)

                if matched_image is not None:
                    gray_matched = cv2.cvtColor(matched_image, cv2.COLOR_BGR2GRAY)
                    keypoints_matched, _ = sift.detectAndCompute(gray_matched, None)

                    # Dibujar los buenos matches
                    img_matches = cv2.drawMatchesKnn(
                        new_image, keypoints, matched_image, keypoints_matched, [[match] for match in best_match_keypoints],
                        None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    # Mostrar la imagen con matplotlib
                    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
                    plt.title(f"En esta imagen hay '{best_match_name}' de la supercategoría '{best_match_supercategory}'")
                    plt.show()
                else:
                    print(f"No se pudo leer la imagen desde el dataset: {image_path}")
            else:
                print(f"No se pudo encontrar la ruta correcta para la imagen: {best_match}")

        else:
            print("No se encontraron buenas coincidencias.")
