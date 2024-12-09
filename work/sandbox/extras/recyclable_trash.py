import json

dataset_path = '../data'
anns_file_path = dataset_path + '/' + 'annotations.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.load(f)

#  Imprime todas las categorias de basura
# for category in dataset['categories']:
#     print(category['name'])

recyclable_trash = [
    "Aluminium foil",
    "Clear plastic bottle",
    "Glass bottle",
    "Metal bottle cap",
    "Aerosol",
    "Drink can",
    "Food Can",
    "Corrugated carton",
    "Egg carton",
    "Toilet tube",
    "Other carton",
    "Glass jar",
    "Metal lid",
    "Normal paper",
    "Wrapping paper",
    "Magazine paper",
    "Paper bag",
    "Pop tab",
    "Scrap metal",
    "Paper straw",
    "Plastic lid",
    "Spread tub"
]

filtered_categories = [
    category for category in dataset['categories'] if category['name'] in recyclable_trash
]

new_json_data = {
    "categories" : filtered_categories
}

with open ('recyclable_categories.json', 'w') as f:
    json.dump(new_json_data, f, indent=4)

print("Nuevo archivo JSON creado con las categorias reciclables.")