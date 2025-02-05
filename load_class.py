import json

def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names