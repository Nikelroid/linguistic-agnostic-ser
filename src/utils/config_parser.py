import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')

def load_config(path=CONFIG_PATH):
    """
    Loads and parses the centralized YAML configuration file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
        
    with open(path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            raise
