import json
import os
import shutil

# Ruta del archivo json filtrado (nuevo JSON con solo reciclables)
filtered_json_path = './filtered_recyclable_annotations.json'

# Directorio donde estan almacenadas las imagenes en las carpetas batch_1 - batch_15
images_root_dir = '../data'

# Directorio donde copiaremos las imagenes filtradas
new_images_dir = './recyclable_images'
os.makedirs(new_images_dir, exist_ok=True)

# Cargar el JSON filtrado
with open(filtered_json_path, 'r') as f:
    filtered_data = json.load(f)
i = 0
# Iterar sobre las imagenes filtradas
for image in filtered_data['images']:
    file_name = image['file_name']

    # Construir la ruta completa de la imagen original
    source_image_path = os.path.join(images_root_dir, file_name)

    # Extraer el nombre de la carpeta batch y la imagen
    batch_folder = os.path.dirname(file_name) 

    # Crear el subdirectorio dentro del nuevo directorio para cada batch
    batch_destination_dir = os.path.join(new_images_dir, batch_folder)
    os.makedirs(batch_destination_dir,exist_ok=True)

    # Construir la nueva ruta en el subdirectorio de la carpeta destino
    destination_image_path = os.path.join(batch_destination_dir, os.path.basename(file_name))

    # Verificar que la imagen exista antes de copiarla
    if os.path.exists(source_image_path):
        # Copiar la imagen al nuevo subdirectorio de destino
        shutil.copy2(source_image_path, destination_image_path)
        print(f'Copiado: {file_name} a {batch_destination_dir}')
        i += 1
    else:
        print(f'Imagen no encontrada: {source_image_path}')
        
print(f"Imagenes copiadas{i}")
print("TODAS LAS IMAGENES RECICLABLES HAN SIDO COPIADAS")