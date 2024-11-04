import yaml
import geojson
import pickle

def load_yaml(path):
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error loading file: {exc}")
        return None
    
def load_pickle(path):
    try:
        with open(path, 'rb') as file:
            loaded_pickle = pickle.load(file)
            return loaded_pickle
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
def save_pickle(path, pickle_to_save):
    with open(path, 'wb') as file:
        pickle.dump(pickle_to_save, file)
        print('File Saved')

def load_geojson(path):
    with open(path) as f:
        data = geojson.load(f)
    return data

def save_geojson(output_filename,feature_collection):
    with open(output_filename, "w") as output_file:
        geojson.dump(feature_collection, output_file)
    print(f"GeoJSON file '{output_filename}' created successfully.")