import torch
import os
from varname import nameof


def get_device():
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def disable_cuda(args=None):
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    if torch.cuda.is_available():
        print("Disable CUDA fail!")
    else:
        print("Disable CUDA success!")
        
        
def set_cuda(gpus=None):
    """_summary_

    Args:
        gpus (int, list): _description_
    """
    
    if gpus == None or gpus == -1:
        disable_cuda()
    else:
        _gpus = []
        if isinstance(gpus, list):
            for g in gpus:
                _gpus.append(str(g))
        elif isinstance(gpus, int):
            _gpus.append(str(gpus))
        else:
            print("Unknow input types!")
            return
            
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join(_gpus)
        
        print("Current CUDA Devices: {}".format(torch.cuda.current_device()))
        print("Total Visible CUDA Device Count: {}".format(torch.cuda.device_count()))
    
def printshape(var):
    print(f"{nameof(var)}: {var.shape}")