from pathlib import Path

import yaml
import h5py
import pickle
import geojson

from toolkit.system.logging_tools import Logger

logger = Logger(name="data_io_tools", log_folder="./logs").get_logger()


class H5:
    def __init__(self):
        pass

    def save_wkt_dict(self, data, path, overwrite=True):
        """
        Saves a dictionary with class names as keys and WKT strings as values to an HDF5 file.

        Parameters:
            data (dict): Dictionary with class names as keys and WKT strings as values.
            path (str or Path): Path to the .h5 file.
        """
        path = Path(path)  # Ensure path is a Path object

        if path.exists():
            if overwrite:
                path.unlink()
            else:
                logger.info(
                    "File already exists, set overwrite=True for overwriting the file."
                )
                return

        with h5py.File(path, "a") as f:
            group = f.create_group(
                "wkt_data"
            )  # Optional: you can save everything inside a group

            for class_name, wkt in data.items():
                group.create_dataset(class_name, data=wkt)

        logger.info(f"h5 file at '{path}' created successfully.")

    def load_wkt_dict(self, path):
        path = Path(path)  # Ensure path is a Path object
        with h5py.File(path, "r") as f:
            data = {}
            group = f["wkt_data"]

            for class_name in group.keys():
                data[class_name] = group[class_name][()]

        return data

    def save_numpy_array(self, array, path, overwrite=False):
        """
        Saves a NumPy array as a dataset in an HDF5 file.

        Parameters:
            array (numpy.ndarray): NumPy to save.
            path (str or Path): Path to the .h5 file.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        path = Path(path)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                logger.info(
                    "File already exists, set overwrite=True for overwriting the file."
                )
                return

        with h5py.File(path, "w") as f:
            f.create_dataset("array", data=array)

    def load_numpy_array(self, path):
        """
        Loads predictions from an HDF5 file.

        Parameters:
            path (str or Path): Path to the .h5 file.

        Returns:
            numpy.ndarray: Loaded NumPy array.
        """
        path = Path(path)

        with h5py.File(path, "r") as f:
            data = f["predictions"][()]

        return data


h5 = H5()


def save_yaml(data, path):
    """
    Saves a dictionary (or any data structure) to a YAML file.

    Parameters:
        data (dict): The data to be saved to the YAML file.
        path (str or Path): The path of the YAML file to save.
    """
    path = Path(path)  # Ensure path is a Path object
    try:
        with open(path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
            logger.info(f"YAML file at '{path}' created successfully.")
    except Exception as e:
        logger.info(f"An error occurred while saving to YAML: {e}")


def load_yaml(path):
    path = Path(path)  # Ensure path is a Path object
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        logger.info(f"Error: The file {path} was not found.")
        return None
    except yaml.YAMLError as exc:
        logger.info(f"Error loading file: {exc}")
        return None


def save_pickle(data, path, replace=False):
    path = Path(path)  # Ensure path is a Path object

    if path.exists() and not replace:
        logger.info(
            f"Pickle alrady exists at {path}, set replace=True, for replacing the file"
        )
    else:
        with open(path, "wb") as file:
            pickle.dump(data, file)
            logger.info(f"Pickle file at '{path}' created successfully.")


def load_pickle(path):
    path = Path(path)  # Ensure path is a Path object
    try:
        with open(path, "rb") as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        logger.info(f"Error: The file {path} was not found.")
        return None
    except Exception as e:
        logger.info(f"Error loading file: {e}")
        return None


def save_geojson(data, path):
    path = Path(path)  # Ensure path is a Path object
    with open(path, "w") as output_file:
        geojson.dump(data, output_file)
    logger.info(f"GeoJSON file at '{path}' created successfully.")


def load_geojson(path):
    path = Path(path)  # Ensure path is a Path object
    with open(path) as f:
        data = geojson.load(f)
    return data
