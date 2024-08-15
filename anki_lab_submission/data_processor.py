import torch
from torchvision.transforms import v2
from torchvision import transforms
import torchvision.models as models
import numpy as np
from trainer import TrainerDistillation, Trainer
import random
import json
from icecream import ic
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
from search_space.RegNet import RegNet
import os
###################

class RandomPixelChange:
    def __init__(self, change_prob=0.1):
        self.change_prob = change_prob
    
    def __call__(self, img):
        # Convert image to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Normalize the array to [0, 1]
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())#
        
        unique_values=np.unique(img_array)
        
        # Generate a random mask with the same shape as img_array
        mask = np.random.rand(*img_array.shape) < self.change_prob
        
        # Apply the mask to randomly change the pixels to any of the unique values
        random_values = np.random.choice(unique_values, size=img_array.shape)
        img_array[mask] = random_values[mask]
        
        return img_array.transpose(1, 2, 0)

#########################
class RandomPixelFlip:
    def __init__(self, flip_prob=0.1):
        self.flip_prob = flip_prob
    
    def __call__(self, img):
        # Convert image to numpy array
        img_array = np.array(img)
        
        min_val=img_array.min()
        max_val=img_array.max()
        
        img_array = (img_array - min_val) / (max_val - min_val)
        
        # Generate a random mask with the same shape as img_array
        mask = np.random.rand(*img_array.shape) < self.flip_prob
        
        # Apply the mask to flip the pixels
        img_array[mask] = 1 - img_array[mask]

        return img_array.transpose(1, 2, 0)

def get_gpu_memory(gpu_id):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    return info.free

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train=False, transform=None, calibration=True):
        self.x = torch.tensor(x)

        # the test dataset has no labels, so we don't need to care about self.y
        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y)


        # example transform
        if train:
            self.mean = torch.mean(self.x, [0, 2, 3])
            self.std = torch.std(self.x, [0, 2, 3])
            self.transform_normalization=[transforms.Normalize(self.mean, self.std)]
            self.transform = transforms.Compose(transform+self.transform_normalization)

        else:
            self.transform=v2.Compose(transform)
            
    

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        im = self.x[idx]

        if self.transform is not None:
            im = self.transform(im)

        # only return image in the case of the test dataloader
        if self.y is None:
            return im
        else:
            return im, self.y[idx]



class DataProcessor:
    """
    -===================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes

    """
    
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, test_y=None):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y=test_y
        self.metadata = metadata
        self.metadata['train_config_path']="../configs/train/augmentations_adam.yaml"
        self.SAVE_PATH="results/full_training_evonas"
        self.multiprocessing=False
        if self.multiprocessing:
            nvmlInit()
            current_method = mp.get_start_method(allow_none=True)
            if current_method!="spawn":
                # Set the start method if it hasn't been set yet
                mp.set_start_method("spawn")

    def process(self):
        """
        ====================================================================================================================
        PROCESS ============================================================================================================
        ====================================================================================================================
        This function will be called, and it expects you to return three outputs:
            * train_loader: A Pytorch dataloader of (input, label) tuples
            * valid_loader: A Pytorch dataloader of (input, label) tuples
            * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!

        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.  

        Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
        """
        # Try different transforms for training, we select the best one and use it
        if "select_augment" in self.metadata:
            train_transform = self._determine_train_transform() if self.metadata["select_augment"] else [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()]                
        else:
            train_transform = self._determine_train_transform()
            #train_transform = []
        
        # Create dataloaders with final transforms
        train_ds = Dataset(self.train_x, self.train_y, train=True, transform=train_transform)
        valid_ds = Dataset(self.valid_x, self.valid_y, train=False, transform=train_ds.transform_normalization)
        test_ds = Dataset(self.test_x, self.test_y, train=False, transform=train_ds.transform_normalization)
        
        batch_size = 128

        # build data loaders
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=batch_size,
                                                   drop_last=True,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)
        return train_loader, valid_loader, test_loader
    
    def _train_model(self, transform, idx, output_queue=None):
            model = models.resnet18(weights=None)
            new_conv1 = torch.nn.Conv2d(in_channels=self.metadata["input_shape"][1], 
                                      out_channels=model.conv1.out_channels, 
                                      kernel_size=model.conv1.kernel_size, 
                                      stride=model.conv1.stride, 
                                      padding=model.conv1.padding, 
                                      bias=model.conv1.bias)
            # Replace the first convolutional layer
            model.conv1 = new_conv1
            model.fc = torch.nn.Linear(512, self.metadata['num_classes'])
            model.to(device)
        
            train_ds = Dataset(self.train_x, self.train_y, train=True, transform=transform)
            valid_ds = Dataset(self.valid_x, self.valid_y, train=False, transform=train_ds.transform_normalization)
            
            # get dataloaders
            batch_size = 128
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
            

            # get trainer and train
            
            self.metadata["experiment_name"]=f"{self.SAVE_PATH}/augmentations_test/{self.metadata['codename']}/aug_{idx}"
            trainer = TrainerDistillation(model, device, train_loader, valid_loader, self.metadata)


            train_acc, val_acc, epoch_time=trainer.train(return_acc=True)        
            
            print(train_acc.cpu().numpy())
            print(val_acc.cpu().numpy())
            
            del trainer, train_loader, valid_loader, train_ds, valid_ds, model, new_conv1
            torch.cuda.empty_cache()
            gc.collect()

            if output_queue is not None:
                output_queue.put((idx, train_acc.cpu().numpy().astype(float).tolist(), val_acc.cpu().numpy().astype(float).tolist(), epoch_time)) 
            else:
                return train_acc.cpu().numpy().astype(float).tolist(), val_acc.cpu().numpy().astype(float).tolist(), epoch_time           


    def _determine_train_transform(self):
        print(self.metadata)
        C,H,W=self.metadata['input_shape'][1:4]
        PH,PW=int(H/8),int(W/8)
        unique_values=np.unique(self.train_x)

        if C==3 and len(unique_values)>3:
            augmentation_combinations = [
                [],  # No augmentation
                [RandomPixelChange(0.02), v2.ToTensor()],
                [RandomPixelChange(0.05), v2.ToTensor()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomCrop((H,W), padding=(PH,PW))],

                [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()],
                [ v2.RandomCrop((H,W), padding=(PH,PW)),
                v2.RandomHorizontalFlip()],
                [v2.RandomGrayscale(p=0.2),v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))],
                [ v2.RandomCrop((H,W), padding=(PH,PW)),
                v2.RandomHorizontalFlip(), v2.RandomGrayscale(p=0.1),v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))],
                               ]
 
        elif C==1 and len(unique_values)>3:
            augmentation_combinations= [
                [],
                [RandomPixelChange(0.02), v2.ToTensor()],
                [RandomPixelChange(0.05), v2.ToTensor()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW)),
                 v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()],       

            ]
        else:
                augmentation_combinations= [
                [],
                [RandomPixelChange(0.01), v2.ToTensor()],
                [RandomPixelChange(0.05), v2.ToTensor()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)), v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW)),v2.RandomHorizontalFlip()]
      
       
            ]
            

        results = {}
        results_val_acc={}
        
        if not self.multiprocessing:
            for idx, transform in enumerate(augmentation_combinations):

                train_acc, val_acc, epoch_time=self._train_model(transform, idx)

                results[str(idx)]={"val_acc":val_acc,
                                   "train_acc":train_acc, 
                                   "epoch_time":epoch_time}
                results_val_acc[str(idx)]=val_acc
        else:
                idx = 0
                output_queue = Queue()
                ic("initial memory")
                print(f"Gpu free memory: {get_gpu_memory(0) / (1024 ** 3):.3f} GB")
                required_memory= 8*2 ** 30
                processes = []
                total_processes_to_run=len(augmentation_combinations)
                while idx < total_processes_to_run:#or any(p.is_alive() for p in processes):
                    if idx<5:
                        sleep_time=3
                    else:
                        sleep_time=10

                    available_memory = get_gpu_memory(0)

                    if (idx < total_processes_to_run) and available_memory>required_memory:

                        p = mp.Process(target=self._train_model, args=(augmentation_combinations[idx], idx, output_queue))
                        p.start()
                        processes.append(p)
                        idx += 1

                        torch.cuda.empty_cache()
                        print(f"Gpu free memory: {available_memory / (1024 ** 3):.3f} GB")
                        ic(idx)
                        
                    time.sleep(sleep_time)  # Sleep for a while before checking again


                get_gpu_memory(0)
                for p in processes:
                    p.join()
                    torch.cuda.empty_cache()

                while not output_queue.empty():
                    idx,train_acc, val_acc, epoch_time=output_queue.get()
                    
                    results[str(idx)]={"val_acc":val_acc,
                       "train_acc":train_acc, 
                       "epoch_time":epoch_time}
                    results_val_acc[str(idx)]=val_acc
                    print(results)
                    print(results_val_acc)

        # save the results to a file
        with open(f"{self.SAVE_PATH}/augmentations_test/{self.metadata['codename']}/augmentation_results.json", 'w') as f:
            json.dump(results, f)

        
        # Sort the dictionary by value in descending order
        sorted_items = sorted(results_val_acc.items(), key=lambda item: item[1], reverse=True)
        print(f"First best key: {sorted_items[0][0]}")
        print(f"Second best key: {sorted_items[1][0]}")
        
        max_key = sorted_items[0][0] if sorted_items[0][0]!="0" else sorted_items[1][0]
        max_value = results_val_acc[max_key]

        print(f'The key with the maximum value is "{max_key}" with a value of {max_value}.')


        return augmentation_combinations[int(max_key)]
