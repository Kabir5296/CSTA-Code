import yaml
from collections import namedtuple

def dict_to_object(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict_to_object(value)
    return namedtuple('CSTAConfig', dictionary.keys())(**dictionary)

def get_config(file_name):
    with open(file_name, "r") as f:
        config=yaml.safe_load(f)
    return dict_to_object(config)