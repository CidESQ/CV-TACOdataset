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

filtered_categories = [
    category for category in dataset['categories'] if category['name'] in recyclable_trash
]

new_json_data = {
    "categories" : filtered_categories
}

with open ('recyclable_categories.json', 'w') as f:
    json.dump(new_json_data, f, indent=4)

print("Nuevo archivo JSON creado con las categorias reciclables.")