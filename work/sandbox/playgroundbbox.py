import os
import cv2
import json
import numpy as np
from pathlib import Path

for a in range(1, 15 + 1):
    # Rutas de directorios
    batch_dir = f"../subset/batch_{a}"  # Directorio donde están las imágenes del batch
    annotations_file = "../subset/subset_annotations.json"  # Archivo de anotaciones en formato COCO
    knowledge_base_dir = "../knowledge_base"  # Carpeta para almacenar los archivos JSON de salida

    # Crear el directorio knowledge_base si no existe
    Path(knowledge_base_dir).mkdir(parents=True, exist_ok=True)

    # Cargar el archivo de anotaciones JSON
    with open(annotations_file, "r") as file:
        annotations = json.load(file)

    # Obtener el mapeo de categorías para un acceso más sencillo
    categories = {category['id']: category for category in annotations['categories']}

    # Inicializar el diccionario para almacenar la información del batch
    batch_data = {}

    # Procesar cada imagen en el directorio batch
    for image_info in annotations['images']:
        if not image_info['file_name'].startswith(f"batch_{a}/"):
            continue

        image_path = os.path.join("../subset", image_info['file_name'])
        image_id = image_info['id']

        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            continue

        # Obtener las anotaciones para la imagen
        annotations_for_image = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        if not annotations_for_image:
            continue

        # Procesar cada bbox en la imagen
        for ann in annotations_for_image:
            # Obtener las coordenadas del bbox
            x, y, w, h = ann['bbox']

            # Recortar el área del bbox
            bbox_region = image[int(y):int(y + h), int(x):int(x + w)]
            if bbox_region.size == 0:
                continue

            # Convertir a escala de grises
            gray = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)

            # Detectar esquinas de Harris
            gray_harris = np.float32(gray)
            harris_corners = cv2.cornerHarris(gray_harris, blockSize=2, ksize=3, k=0.04)
            harris_corners = cv2.dilate(harris_corners, None)

            # Normalizar la imagen de Harris para usarla con SIFT
            harris_image_uint8 = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Usar SIFT para obtener los descriptores
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(harris_image_uint8, None)

            # Extraer los campos supercategory y name de la categoría a la que pertenece el bbox
            category_info = categories[ann['category_id']] if 'category_id' in ann else None
            supercategory = category_info['supercategory'] if category_info else "Unknown"
            name = category_info['name'] if category_info else "Unknown"

            # Preparar los datos a guardar en el JSON
            file_name = os.path.basename(image_info['file_name'])

            # Verificar si la imagen ya existe en el diccionario y agregar los descriptores del bbox a la lista
            if file_name not in batch_data:
                batch_data[file_name] = {
                    "image_id": image_id,
                    "bboxes": []  # Lista para almacenar información de los bboxes
                }

            if descriptors is not None:
                bbox_info = {
                    "bbox": ann['bbox'],
                    "supercategory": supercategory,
                    "name": name,
                    "descriptors": descriptors.tolist()
                }
                batch_data[file_name]["bboxes"].append(bbox_info)

    # Guardar los resultados en un archivo JSON dentro de knowledge_base
    output_file = os.path.join(knowledge_base_dir, f"descriptores_bbox_batch{a}.json")
    with open(output_file, 'w') as outfile:
        json.dump(batch_data, outfile, indent=4)

    print(f"Resultados guardados en: {output_file}")
    print("Procesamiento finalizado.")
    