import json
import os 
MAIN_DIR = os.path.split(os.path.dirname(__file__))[0]
CONFIGS_FILE = os.path.join(MAIN_DIR, 'config.json')

with open(CONFIGS_FILE, "r") as f:
    configs = json.load(f)