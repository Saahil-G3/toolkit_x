import tensorflow as tf

def check_dl_framework_detections():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"{len(tf.config.experimental.list_physical_devices('GPU'))} devices detected.")
        for gpu in gpus:
            print(gpu)
    else:
        print("No GPUs detected.")