import sys
import torch
import random
import numpy as np
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import gc
from icecream import ic

sys.path.append("../")
####### Dataset ############
from evaluation import main
from data_processor import DataProcessor
######## Search space #########
from search_space.RegNet import RegNet
from search_space.utils import create_widths_plot, scatter_results, get_generation_dfs
######## Training ###############
from trainer import Trainer, TrainerDistillation
from utils.train_cfg import get_cfg, show_cfg
###################################################
random_seed = 1
random.seed(random_seed)
# Set seed for NumPy
np.random.seed(random_seed)
# Set seed for PyTorch
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# Additional steps if using CuDNN (optional, for GPU acceleration)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from coolname import generate_slug
from sklearn.metrics import accuracy_score
import torch.multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nvmlInit()

# Function to get available memory for a specific GPU
def get_gpu_memory(gpu_id):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"Gpu free memory: {info.free / (1024 ** 3):.3f} GB")
    return info.free

def measure_memory_usage(model, device):
    torch.cuda.reset_peak_memory_stats(device)
    start=get_gpu_memory(0)
    Dataset="AddNIST"
    (train_x, train_y), (valid_x, valid_y), (test_x), metadata = main.load_datasets(Dataset, truncate=True)
    test_y = np.load(os.path.join('../datasets/'+Dataset,'test_y.npy'))
    metadata["rand_augment"]=False
    data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata)
    train_loader, valid_loader, test_loader = data_processor.process()
    # Reset peak memory stats

    metadata["train_config_path"]="../configs/train/vanilla_generation.yaml"
    train_cfg=get_cfg()
    train_cfg.merge_from_file(metadata["train_config_path"])
    metadata["experiment_name"]=f"{test_folder}/memory"    
    trainer=TrainerDistillation(model, device, train_loader, valid_loader, metadata) 
    trainer.train()
    end=get_gpu_memory(0)
    # Measure the peak memory usage
    max_memory=torch.cuda.max_memory_allocated(device)
    ic(f"Measured peak memory usage: {max_memory / (1024 ** 3):.2f} GB") 
    ic(f"Measured peak memory usage  2: {(start-end) / (1024 ** 3):.2f} GB") 
    del trainer, train_loader, valid_loader, test_loader, data_processor, metadata, train_x,train_y,valid_x,valid_y,test_x,train_cfg, model, device
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    return max_memory

def train_mp(model,test_folder,model_name,metadata,device,train_loader,valid_loader):
    clear_output(wait=True)
    metadata["experiment_name"]=f"{test_folder}/Generation_1/{model_name}"
    trainer=TrainerDistillation(model, device, train_loader, valid_loader, metadata) 
    trainer.train()
    torch.cuda.empty_cache()
    del trainer, train_loader, valid_loader, metadata, model, device

    gc.collect()


    
if __name__ == '__main__':
    
    #Dataset must be in the folder datasets
    # From https://github.com/Towers-D/NAS-Unseen-Datasets
    Dataset="AddNIST"
    (train_x, train_y), (valid_x, valid_y), (test_x), metadata = main.load_datasets(Dataset, truncate=False)
    test_y = np.load(os.path.join('../datasets/'+Dataset,'test_y.npy'))
    metadata["rand_augment"]=False
    data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata)
    train_loader, valid_loader, test_loader = data_processor.process()

    #metadata={"num_classes": 4, "input_shape": [49260, 3, 64, 64], "codename": "Caitie", "benchmark":47.008}
    regnet_space=RegNet(metadata,
                        W0=[16, 64, 8],
                        WA=[8, 48, 8],
                        WM=[2.05,2.9,0.05],
                        D=[8,22,1], 
                        G=[8,16,8], 
                        base_config="../configs/search_space/config.yaml")


    
    # Folder to save the model
    test_folder="test_multiprocessing"
    # Config update is used to modify other parameter from the config
    #config_updates=["REGNET.SE_ON", False]
    #config_updates=["REGNET.DROP_RATE",0.05, "REGNET.DROPOUT",0.1]


    models, chromosomes=regnet_space.create_random_generation(save_folder=test_folder,gen=1, size=10, config_updates=None)
    # Show train config file
    metadata["train_config_path"]="../configs/train/vanilla_generation_lion.yaml"
    train_cfg=get_cfg()
    train_cfg.merge_from_file(metadata["train_config_path"])
    
    
    
    mp.set_start_method('spawn')
    ###################################
    num_initial_processes = 0
    ic(num_initial_processes)
    names_list=list(models.keys())
    total_processes_to_run = len(names_list) 
    
    #required_memory={}

    required_memory= 2*2 ** 30

    available_memory = get_gpu_memory(0)

    
    start_time=time.time()
    
    
    #for index in range(num_initial_processes):
    #    key=names_list[index]
    #    p = mp.Process(target=train_mp, args=(models[key],test_folder,key,metadata,device,train_loader,valid_loader))
    #    p.start()
    #    processes.append(p)
    multi=False
    
    if not multi:
        for model_name, model in models.items():
            train_mp(model,test_folder,model_name,metadata,device,train_loader,valid_loader)
    else:  
        processes = []
        next_process_index = num_initial_processes

        ic("initial memory")
        get_gpu_memory(0)
        #while all(p.is_alive() for p in processes) and len(processes)!=0:
        #    time.sleep(10)

        while next_process_index < total_processes_to_run:#or any(p.is_alive() for p in processes):
            if next_process_index<10:
                sleep_time=1
            else:
                sleep_time=10

            available_memory = get_gpu_memory(0)
            key=names_list[next_process_index]

            if (next_process_index < total_processes_to_run) and available_memory>required_memory:
                p = mp.Process(target=train_mp, args=(models[key],test_folder,key,metadata,device,train_loader,valid_loader))
                p.start()
                processes.append(p)
                next_process_index += 1
                ic(next_process_index)
                ic(key)

            time.sleep(sleep_time)  # Sleep for a while before checking again
            ic(len(processes))


        get_gpu_memory(0)
        for p in processes:
            p.join()
        
    end_time=time.time()
    ic((end_time-start_time))