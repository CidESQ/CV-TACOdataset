import json

# IDs de categorías reciclables en el dataset TACO
recyclable_categories = {0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 52}

# Cargar el archivo JSON
with open('annotations.json', 'r') as file:
    data = json.load(file)

# Inicializar diccionarios para guardar las imágenes y anotaciones de basura reciclable
recyclable_images = []
recyclable_annotations = []

# Recorrer las anotaciones y filtrar las que corresponden a categorías reciclables
for annotation in data['annotations']:
    if annotation['category_id'] in recyclable_categories:
        recyclable_annotations.append(annotation)
        image_id = annotation['image_id']

        # Buscar la imagen correspondiente al 'image_id' y añadirla a la lista si no está ya
        if not any(image['id'] == image_id for image in recyclable_images):
            image = next(img for img in data['images'] if img['id'] == image_id)
            recyclable_images.append(image)

# Crear un nuevo archivo JSON con las imágenes y anotaciones reciclables
recyclable_data = {
    'info': data['info'],
    'images': recyclable_images,
    'annotations': recyclable_annotations,
    'categories': [cat for cat in data['categories'] if cat['id'] in recyclable_categories]
}

# Guardar el nuevo archivo JSON
with open('recyclable_annotations.json', 'w') as outfile:
    json.dump(recyclable_data, outfile, indent=4)

print("Imágenes y anotaciones de basura reciclable guardadas en 'recyclable_annotations.json'.")
