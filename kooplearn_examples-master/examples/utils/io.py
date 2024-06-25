from utils.typing import JsonNameSpace
import json
import haiku as hk
import pickle

def save_haiku_checkpoint(params:hk.Params, path:str):
    with open(path, 'wb') as fp:
        pickle.dump(params, fp)

def load_haiku_checkpoint(path:str) -> hk.Params:
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def load_json(path:str) -> JsonNameSpace:
    "A simple utility to parse json files to Python objects accessible via dot notation"
    with open(path, "r") as json_file:
        x = json.load(json_file, object_hook=lambda x: JsonNameSpace(**x))
    return x