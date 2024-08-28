import sys
import ast
import torch
import random
import numpy as np
import time
import os
from IPython.display import clear_output
sys.path.append("anki_lab_submission")
####### Dataset ############
from data_processor import DataProcessor
######## Search space #########
from search_space.RegNet import RegNet
from search_space.utils import create_widths_plot, scatter_results, get_generation_dfs
######## Training ###############
from trainer import TrainerDistillation
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
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from datetime import datetime
import itertools
import pandas as pd
import json
from io import StringIO
from coolname import generate_slug
from sklearn.metrics import accuracy_score
import torch.multiprocessing as mp
from icecream import ic
import gc


import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo




def get_gpu_memory(gpu_id):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    return info.free


def load_dataset_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'metadata'), "r") as f:
        metadata = json.load(f)
    return metadata

# load dataset from file
def load_datasets(data_path, truncate):
    data_path = 'datasets/'+data_path
    train_x = np.load(os.path.join(data_path,'train_x.npy'))
    train_y = np.load(os.path.join(data_path,'train_y.npy'))
    valid_x = np.load(os.path.join(data_path,'valid_x.npy'))
    valid_y = np.load(os.path.join(data_path,'valid_y.npy'))
    test_x = np.load(os.path.join(data_path,'test_x.npy'))
    metadata = load_dataset_metadata(data_path)

    if truncate:
        train_x = train_x[:64]
        train_y = train_y[:64]
        valid_x = valid_x[:64]
        valid_y = valid_y[:64]
        test_x = test_x[:64]

    return (train_x, train_y), \
           (valid_x, valid_y), \
           (test_x), metadata

def validation(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 4. Calculate Accuracy
    accuracy = correct / total
    print('Accuracy on the test set: {:.2f}%'.format(accuracy * 100))

def train_mp(student,student_name, metadata, test_folder, device, train_loader,valid_loader):
        
        clear_output(wait=True)
        metadata["experiment_name"]=f"{test_folder}/{student_name}"
        trainer=TrainerDistillation(student, device, train_loader, valid_loader,metadata) 
        trainer.train()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    current_method = mp.get_start_method(allow_none=True)
    print(current_method)
    if current_method!="spawn":
        nvmlInit()
        # Set the start method if it hasn't been set yet
        mp.set_start_method("spawn")
    SUBMISSION_PATH="anki_lab_submission"
    Dataset="Language"
    (train_x, train_y), (valid_x, valid_y), (test_x), metadata = load_datasets(Dataset, truncate=False)
    test_y = np.load(os.path.join('datasets/'+Dataset,'test_y.npy'))
    metadata["select_augment"]=False
    data_processor = DataProcessor(train_x[:], train_y[:], valid_x, valid_y, test_x, metadata)
    train_loader, valid_loader, test_loader = data_processor.process()
    
    rg=RegNet(metadata,
                    W0=[16, 120, 8],
                    WA=[16, 64, 8],
                    WM=[2.05,2.9,0.05],
                    D=[8,22,1], 
                    G=[8,8,8], 
                    base_config=f"{SUBMISSION_PATH}/configs/search_space/config.yaml")

    current_time=datetime.now().strftime("%d_%m_%Y_%H_%M")
    test_folder=f"{os.getenv('WORK')}/NAS_COMPETITION_RESULTS/kwnowledge_distillation/vanilla_hs/{current_time}/{metadata['codename']}"

    folder=f"/home/woody/iwb3/iwb3021h/NAS_COMPETITION_RESULTS/classifier_train/{metadata['codename']}"
    models, chromosomes=rg.load_generation(folder)
    #models, chromosomes=rg.create_random_generation(save_folder=test_folder,gen=None, size=3, config_updates=None)
    
    # Train models
    metadata["train_config_path"]=f'{SUBMISSION_PATH}/configs/train/first_generation_adam.yaml'
    train_cfg=get_cfg()
    train_cfg.merge_from_file(metadata["train_config_path"])
    
    os.makedirs(test_folder, exist_ok=True)
    output_file_path = f"{test_folder}/config.yaml"
    with open(output_file_path, "w") as f:
            f.write(train_cfg.dump()) 

    models_names=sorted(list(models.keys()))[:] 
    multi=False
    ic((get_gpu_memory(0) / (1024 ** 3)))
    if multi:
        #WITH MULTIPROCESSING
        next_process_index = 0
        ic("initial memory")
        print(f"Gpu free memory: {get_gpu_memory(0) / (1024 ** 3):.3f} GB")
        required_memory= 5*2 ** 30

        processes = []
        total_processes_to_run=len(models_names)
        while next_process_index < total_processes_to_run:#or any(p.is_alive() for p in processes):
            if next_process_index<5:
                sleep_time=3
            else:
                sleep_time=10

            available_memory = get_gpu_memory(0)

            if (next_process_index < total_processes_to_run) and available_memory>required_memory:
                student=models_names[next_process_index]
                p = mp.Process(target=train_mp, args=(models[student],student, metadata, test_folder, device, train_loader,valid_loader))
                p.start()
                processes.append(p)
                next_process_index += 1
                print(f"Gpu free memory: {available_memory / (1024 ** 3):.3f} GB")
                ic(next_process_index)
                ic(student)

            time.sleep(sleep_time)  # Sleep for a while before checking again
            new_avail_mem=get_gpu_memory(0)
            if (new_avail_mem-available_memory)>required_memory:
                required_memory=new_avail_mem-available_memory

        get_gpu_memory(0)
        for p in processes:
            p.join()
    else:
         for name in models_names[:50]:
                train_mp(models[name],name, metadata, test_folder, device, train_loader,valid_loader)



    results_df=get_generation_dfs(f"{test_folder}", corr=False, chromosomes=chromosomes, save=True, gen=None)