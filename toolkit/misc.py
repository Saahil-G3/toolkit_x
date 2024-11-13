import sys
import os
import time
import numpy as np
from tqdm.auto import tqdm
import subprocess
import random
import string
from pathlib import Path

def get_dirs(prefix, project_name):

    data_root = f"{prefix}/data"
    code_root = f"{prefix}/code"

    data_folder = Path(f"{data_root}/{project_name}")
    code_folder = Path(f"{code_root}/{project_name}")

    data_folder.mkdir(exist_ok=True)
    code_folder.mkdir(exist_ok=True)

    
    return data_folder, code_folder

def generate_random_code(length=6):
    # Define the character set to choose from: letters and digits
    characters = string.ascii_letters + string.digits
    # Generate a random code
    return ''.join(random.choice(characters) for _ in range(length))

def get_threads(percent):
    """
    """
    if(int(percent)>100):
        print('percentage should be less than 100')
        return
    
    percent = percent/100
    total_cores = os.cpu_count()
    threads = int(percent*total_cores)
    print(f'Using {threads} threads')
    return threads

def clear_output():
    try:
        # Try to clear Jupyter notebook output
        from IPython.display import clear_output as jupyter_clear_output
        jupyter_clear_output(wait=True)
    except ImportError:
        # If not in Jupyter, try to clear terminal output
        print("\033c", end="")  # This clears the terminal screen (works on Unix-like systems)

def round_to_nearest_even(x):
    return round(x / 2) * 2
    
def start_timer(return_time =False):
    
    """
    Example Usage:
    
    start_time = u.start_timer()
    lap = u.stop_timer()
    
    """
    
    global start_time
    start_time = time.time()
    
    if return_time:
        return({'start_time':start_time})
    
def stop_timer(return_time=False, print_total_time=True):
    """
    Returns : Total time elapsed since last start_timer(), Current time
    """ 
    total_time = time.time()-start_time

    if print_total_time:
        print(f'Total time taken: {round(total_time/60,2)} mins')
    
    if return_time:
        return({'total_time': round(total_time/60,2)})


def get_memory_occupied(python_object):
    memory_usage=sys.getsizeof(python_object) / (1024 ** 2)
    print(f'Size occupied: {round(memory_usage, 2)} MB')