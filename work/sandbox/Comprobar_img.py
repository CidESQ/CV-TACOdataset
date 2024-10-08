import os

# Ruta de la carpeta principal donde se encuentran las carpetas batch
new_images_dir = '../recyclable_images'  # Cambia esta ruta según donde estén tus carpetas batch

# Variable para almacenar el número total de archivos
total_files = 0

# Obtener todas las carpetas batch dentro de `new_images_dir`
batch_folders = [f.path for f in os.scandir(new_images_dir) if f.is_dir()]

# Iterar sobre cada carpeta batch
for batch_folder in batch_folders:
    # Obtener la lista de archivos en la carpeta batch actual
    files = [f for f in os.listdir(batch_folder) if os.path.isfile(os.path.join(batch_folder, f))]
    
    # Contar los archivos en la carpeta actual y sumarlos al total
    total_files += len(files)
    print(f"Carpeta {batch_folder} contiene {len(files)} archivos.")

# Imprimir el total de archivos
print(f"El número total de archivos en todas las carpetas batch es: {total_files}")
