qc_models_v1_args = {}
 
node_detection_v1 = {}
node_detection_v1["gpu_id"] = 0
node_detection_v1["patch_size"] = 256
node_detection_v1["device_type"] = "gpu"
node_detection_v1["dataparallel"] = None
qc_models_v1_args["node_detection_v1"] = node_detection_v1

tissue_model_v1 = {}
tissue_model_v1["gpu_id"] = 0
tissue_model_v1["patch_size"] = 256
tissue_model_v1["device_type"] = "gpu"
tissue_model_v1["dataparallel"] = None
qc_models_v1_args["tissue_model_v1"] = tissue_model_v1

folds_model_v1 = {}
folds_model_v1["gpu_id"] = 0
folds_model_v1["patch_size"] = 256
folds_model_v1["device_type"] = "gpu"
folds_model_v1["dataparallel"] = None
qc_models_v1_args["folds_model_v1"] = folds_model_v1

focus_model_v1 = {}
focus_model_v1["gpu_id"] = 0
focus_model_v1["patch_size"] = 256
focus_model_v1["device_type"] = "gpu"
focus_model_v1["dataparallel"] = None
qc_models_v1_args["focus_model_v1"] = focus_model_v1

pen_model_v1 = {}
pen_model_v1["gpu_id"] = 0
pen_model_v1["patch_size"] = 256
pen_model_v1["device_type"] = "gpu"
pen_model_v1["dataparallel"] = None
qc_models_v1_args["pen_model_v1"] = pen_model_v1