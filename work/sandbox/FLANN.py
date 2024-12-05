import os
import json
import cv2
import numpy as np

# Ruta a la base de conocimiento
knowledge_base_dir = "../knowledge_base"

# Cargar todos los descriptores de la base de conocimiento
knowledge_data = []
for file_name in os.listdir(knowledge_base_dir):
    if file_name.startswith("descriptores_batch") or file_name.startswith("descriptores_bbox_batch"):
        with open(os.path.join(knowledge_base_dir, file_name), "r") as file:
            knowledge_data.append(json.load(file))

# Configuración del FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Leer la nueva imagen que se quiere clasificar
new_image_path = "../test_images/000012.jpg"  #! Cambia esta ruta a la imagen que desees clasificar
new_image = cv2.imread(new_image_path)

if new_image is None:
    print(f"No se pudo leer la imagen: {new_image_path}")
else:
    # Convertir a escala de grises
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Detectar esquinas de Harris
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
        best_match_count = 0

        # Realizar matching con los descriptores de la base de conocimiento
        for data in knowledge_data:
            for image_name, image_info in data.items():
                for bbox_info in image_info.get("bboxes", [{"descriptors": image_info["descriptors"]}]):
                    # Obtener los descriptores de la base de conocimiento
                    knowledge_descriptors = np.array(bbox_info["descriptors"], dtype=np.float32)
                    
                    # Hacer matching con FLANN
                    matches = flann.knnMatch(descriptors, knowledge_descriptors, k=2)

                    # Aplicar la prueba de ratio de Lowe para filtrar buenas coincidencias
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                    # Actualizar la mejor coincidencia si es necesario
                    if len(good_matches) > best_match_count:
                        best_match_count = len(good_matches)
                        best_match = {
                            "image_name": image_name,
                            "category": image_info.get("categories", [{"name": "Unknown"}])[0]['name'],
                            "match_count": len(good_matches)
                        }

        # Mostrar el resultado de la mejor coincidencia
        if best_match:
            print(f"La mejor coincidencia es con la imagen '{best_match['image_name']}' de la categoría '{best_match['category']}' con {best_match['match_count']} coincidencias.")
        else:
            print("No se encontraron buenas coincidencias.")

