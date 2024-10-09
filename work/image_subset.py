import json
import os
import shutil

# Definir las rutas de los archivos y directorios
ruta_json_filtrado = './recyclable_annotations.json'
directorio_imagenes = '../data'
directorio_destino = './imagenes_reciclables'

# Crear el directorio de destino si no existe
os.makedirs(directorio_destino, exist_ok=True)

# Cargar datos desde el archivo JSON filtrado
with open(ruta_json_filtrado, 'r') as archivo_json:
    datos_filtrados = json.load(archivo_json)

# Contador de imágenes copiadas
contador_imagenes = 0

# Procesar cada imagen listada en el JSON
for imagen in datos_filtrados['images']:
    nombre_archivo = imagen['file_name']

    # Determinar la ruta completa de la imagen original
    ruta_imagen_origen = os.path.join(directorio_imagenes, nombre_archivo)

    # Obtener la carpeta batch del archivo
    carpeta_batch = os.path.dirname(nombre_archivo)

    # Crear el subdirectorio correspondiente en el directorio de destino
    ruta_subdirectorio_destino = os.path.join(directorio_destino, carpeta_batch)
    os.makedirs(ruta_subdirectorio_destino, exist_ok=True)

    # Definir la ruta de destino para copiar la imagen
    ruta_imagen_destino = os.path.join(ruta_subdirectorio_destino, os.path.basename(nombre_archivo))

    # Verificar si la imagen existe antes de copiarla
    if os.path.exists(ruta_imagen_origen):
        shutil.copy2(ruta_imagen_origen, ruta_imagen_destino)
        print(f'Imagen copiada: {nombre_archivo} -> {ruta_subdirectorio_destino}')
        contador_imagenes += 1
    else:
        print(f'Imagen no encontrada: {ruta_imagen_origen}')

# Mostrar el total de imágenes copiadas
print(f'Se han copiado {contador_imagenes} imágenes.')
print("PROCESO FINALIZADO: Todas las imágenes reciclables han sido transferidas.")
