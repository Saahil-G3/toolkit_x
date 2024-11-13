from pathlib import Path
script_dir = Path(__file__).parent
weights_dir = script_dir/"weights"

def get_metadata():
    return _metadata.copy()

"""
_metadata["new_model"] = {}
_metadata["new_model"]["path"] = ""  # Path to the model weights
_metadata["new_model"]["class_map"] = {}  # Empty class map
_metadata["new_model"]["mpp"] = ""  # MPP value for this model
_metadata["new_model"]["model_config"] = {}
_metadata["new_model"]["model_config"]["architecture"] = ""  # Model architecture
_metadata["new_model"]["model_config"]["encoder_name"] = ""  # Encoder type
_metadata["new_model"]["model_config"]["encoder_weights"] = ""  # Encoder weights
_metadata["new_model"]["model_config"]["in_channels"] = ""  # Number of input channels
_metadata["new_model"]["model_config"]["classes"] = ""  # Number of classes for this model

"""

_metadata = {}

_metadata["tissue"] = {}
_metadata["tissue"]["path"] = Path(f"{weights_dir}/tissue.pt")
_metadata["tissue"]["class_map"] = {"bg": 0, "adipose": 1, "non_adipose": 2}
_metadata["tissue"]["mpp"] = 4
_metadata["tissue"]["model_config"] = {}
_metadata["tissue"]["model_config"]["architecture"] = "UnetPlusPlus"
_metadata["tissue"]["model_config"]["encoder_name"] = "resnet18"
_metadata["tissue"]["model_config"]["encoder_weights"] = "imagenet"
_metadata["tissue"]["model_config"]["in_channels"] = 3
_metadata["tissue"]["model_config"]["classes"] = 3
_metadata["tissue"]["results_path"] = {"h5": None, "geojson": None}

_metadata["folds"] = {}
_metadata["folds"]["path"] = Path(f"{weights_dir}/folds.pt")
_metadata["folds"]["class_map"] = {"bg": 0, "fold": 1}
_metadata["folds"]["mpp"] = 2
_metadata["folds"]["model_config"] = {}
_metadata["folds"]["model_config"]["architecture"] = "UnetPlusPlus"
_metadata["folds"]["model_config"]["encoder_name"] = "resnet18"
_metadata["folds"]["model_config"]["encoder_weights"] = "imagenet"
_metadata["folds"]["model_config"]["in_channels"] = 3
_metadata["folds"]["model_config"]["classes"] = 2
_metadata["folds"]["results_path"] = {"h5": None, "geojson": None}

_metadata["focus"] = {}
_metadata["focus"]["path"] = Path(f"{weights_dir}/focus.pt")
_metadata["focus"]["class_map"] = {
    "bg": 0,
    "level_1": 1,
    "level_2": 2,
    "level_3": 3,
    "level_4": 4,
}
_metadata["focus"]["mpp"] = 2
_metadata["focus"]["model_config"] = {}
_metadata["focus"]["model_config"]["architecture"] = "UnetPlusPlus"
_metadata["focus"]["model_config"]["encoder_name"] = "resnet18"
_metadata["focus"]["model_config"]["encoder_weights"] = "imagenet"
_metadata["focus"]["model_config"]["in_channels"] = 3
_metadata["focus"]["model_config"]["classes"] = 5
_metadata["focus"]["results_path"] = {"h5": None, "geojson": None}

_metadata["pen"] = {}
_metadata["pen"]["path"] = Path(f"{weights_dir}/pen.pt")
_metadata["pen"]["class_map"] = {"bg": 0, "pen_mark": 1}
_metadata["pen"]["mpp"] = 16
_metadata["pen"]["model_config"] = {}
_metadata["pen"]["model_config"]["architecture"] = "UnetPlusPlus"
_metadata["pen"]["model_config"]["encoder_name"] = "resnet34"
_metadata["pen"]["model_config"]["encoder_weights"] = "imagenet"
_metadata["pen"]["model_config"]["in_channels"] = 3
_metadata["pen"]["model_config"]["classes"] = 2
_metadata["pen"]["results_path"] = {"h5": None, "geojson": None}
