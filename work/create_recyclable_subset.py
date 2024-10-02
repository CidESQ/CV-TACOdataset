import json

# Ruta del archivo original
dataset_path = '../data'
anns_file_path = dataset_path + '/' + 'annotations.json'

# Definir las categorías reciclables
recyclable_trash = [
    "Aluminium foil",
    "Aluminium blister pack",
    "Clear plastic bottle",
    "Glass bottle",
    "Food Can",
    "Aerosol",
    "Drink can",
    "Other carton",
    "Egg carton",
    "Drink carton",
    "Corrugated carton",
    "Meal carton",
    "Pizza box",
    "Normal paper",
    "Paper bag",
    "Plastic film",
    "Polypropylene bag",
    "Crisp packet",
    "Spread tub",
    "Tupperware",
    "Plastic utensils",
    "Plastic straws",
    "Paper straws",
    "Scrap metal"
]

# Cargar el archivo JSON original
with open(anns_file_path, 'r') as f:
    dataset = json.load(f)

# Filtrar las categorías reciclables
recyclable_categories = [
    category for category in dataset['categories'] if category['name'] in recyclable_trash
]

# Obtener los IDs de las categorías reciclables
recyclable_category_ids = {category['id'] for category in recyclable_categories}

# Filtrar las anotaciones relacionadas con categorías reciclables
filtered_annotations = [
    annotation for annotation in dataset['annotations']
    if annotation['category_id'] in recyclable_category_ids
]

# Obtener los IDs de las imágenes a partir de las anotaciones filtradas
filtered_image_ids = {annotation['image_id'] for annotation in filtered_annotations}

# Filtrar las imágenes relacionadas con las anotaciones filtradas
filtered_images = [
    image for image in dataset['images']
    if image['id'] in filtered_image_ids
]

# Mantener la información adicional
new_json_data = {
    "info": dataset["info"],
    "images": filtered_images,
    "annotations": filtered_annotations,
    "scene_annotations": dataset.get("scene_annotations", []),  # Asumiendo que puede no existir
    "licenses": dataset.get("licenses", []),  # Asumiendo que puede no existir
    "categories": recyclable_categories,
    "scene_categories": dataset.get("scene_categories", [])  # Asumiendo que puede no existir
}

# Guardar el nuevo archivo JSON
with open('filtered_recyclable_annotations.json', 'w') as f:
    json.dump(new_json_data, f, indent=4)

print("Nuevo archivo JSON creado con la información sobre la basura reciclable.")
