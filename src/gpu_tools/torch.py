import torch
import subprocess

def get_device(gpu_id):
    """
    Get a specific GPU device based on the provided gpu_id.

    Args:
    gpu_id (int): The ID of the GPU device (e.g., 0 for the first GPU, 1 for the second GPU).

    Returns:
    torch.device: The PyTorch device object for the specified GPU.
    """
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU ID {gpu_id}. Available GPU IDs: 0 to {torch.cuda.device_count() - 1}.")
        return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')
        #raise RuntimeError("CUDA is not available. No GPU devices found.")
        

def check_dl_framework_detections():
        if torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} devices detected.")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No GPUs detected.")
        

def get_gpu_memory_info():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8'
        )
        print("GPU Memory Information:\n------------------------")
        for i, line in enumerate(result.strip().split('\n')):
            free_memory, total_memory = line.split(',')
            free_memory = float(free_memory)
            total_memory = float(total_memory)
            percent_free =  round((free_memory/total_memory)*100, 2)
            
            print(f"GPU {i}:\n Free: {percent_free} %, {round(free_memory/1024, 2)} GB")
            print(f" Occupied: {round(100-percent_free, 2)} %, {round((total_memory-free_memory)/1024, 2)} GB")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
                
