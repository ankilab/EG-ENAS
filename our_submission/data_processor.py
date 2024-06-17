import torch
from torchvision.transforms import v2
import torchvision.models as models
import numpy as np
from utils.transforms import get_train_transform
from trainer import TrainerDistillation
# import torchvision.transforms as transforms
import json
import random
from icecream import ic
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
from search_space.RegNet import RegNet

def get_gpu_memory(gpu_id):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    return info.free

class AddNoise:
    def __init__(self, probability=0.1, mean=0.0, stddev=1.0):
        self.probability = probability
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        # Create a mask with the same shape as the tensor, where each element has a specified probability of being 1 (adding noise)
        mask = torch.rand(tensor.shape) < self.probability

        # Generate Gaussian noise with the same shape as the tensor
        noise = torch.randn(tensor.shape) * self.stddev + self.mean

        # Apply the noise to the tensor based on the mask
        noisy_tensor = torch.where(mask, tensor + noise, tensor)

        return noisy_tensor
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train=False, transform=None, calibration=True):
        self.x = torch.tensor(x)

        # the test dataset has no labels, so we don't need to care about self.y
        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y)

        print(transform)

        # example transform
        if train:
            self.mean = torch.mean(self.x, [0, 2, 3])
            self.std = torch.std(self.x, [0, 2, 3])
            self.transform_normalization=[v2.Normalize(self.mean, self.std)]
            #print(self.transform_normalization)
            #if calibration:
            # print(transform)
            self.transform = v2.Compose(transform+self.transform_normalization)
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

class SaltAndPepperNoise:
    def __init__(self, probability=0.05, salt_ratio=0.5):
        self.probability = probability
        self.salt_ratio = salt_ratio

    def __call__(self, tensor):
        noisy_tensor = tensor.clone()
        max_value = torch.max(tensor)
        min_value = torch.min(tensor)

        # Generate random mask for salt and pepper noise
        salt_mask = torch.rand(tensor.shape) < self.probability / 2
        pepper_mask = torch.rand(tensor.shape) < self.probability / 2

        # Add salt noise
        noisy_tensor[salt_mask] = max_value  # Set salt noise to maximum value

        # Add pepper noise
        noisy_tensor[pepper_mask] = min_value  # Set pepper noise to minimum value

        return noisy_tensor


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
        self.metadata['train_config_path']="configs/train/augmentations_adam.yaml"
        self.metadata["experiment_name"]="tests/augmentations_test"
        self.multiprocessing=True
        if self.multiprocessing:
            nvmlInit()
            current_method = mp.get_start_method(allow_none=True)
            #print(current_method)
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
        
        train_transform = self._determine_train_transform() 
        
        
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
    
    def _train_model(self,model, transform, idx, output_queue=None):
            train_ds = Dataset(self.train_x, self.train_y, train=True, transform=transform)
            valid_ds = Dataset(self.valid_x, self.valid_y, train=False, transform=train_ds.transform_normalization)

            #study_folder="tests_LaMelo_13_06_2024_19_41"
            #regnet_space=RegNet(self.metadata,
            #        W0=[16, 96, 8],
            #        WA=[16, 64, 8],
            #        WM=[2.05,2.9,0.05],
            #        D=[8,22,1], 
            #        G=[8,8,8], 
            #        base_config=f"../configs/search_space/config.yaml")
            #test_folder=f"/home/woody/iwb3/iwb3021h/NAS_COMPETITION_RESULTS/full_training/{study_folder}"
            #gen=1
            #model_name="mutant_spoonbill"
            #config_updates=["REGNET.DROP_RATE",0.05, "REGNET.DROPOUT",0.1]
            #config_updates=None
            # If already trained, add weights_file.
            #weights_file=f"{test_folder}/Generation_{gen}/{model_name}/student_best"
            #weights_file=f"../../package/tests/Sadie/finetuning/14_06_2024_06_58_adam/student_best"
            #weights_file=None
            #model,info=regnet_space.load_model(config_file=f"{test_folder}/Generation_{gen}/{model_name}/config.yaml",
            #                               weights_file=weights_file)

            
            # get ResNet-18 model


            # get dataloaders
            batch_size = 128
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

            # get trainer and train
            self.metadata["experiment_name"]=f"tests/augmentations_test/{self.metadata['codename']}/aug_{idx}"
            trainer = TrainerDistillation(model, device, train_loader, valid_loader, self.metadata)
            #trainer.epochs = 10
            train_acc, val_acc, epoch_time=trainer.train(return_acc=True)
            #acc = trainer.evaluate()
            print(train_acc.cpu().numpy())
            print(val_acc.cpu().numpy())
            torch.cuda.empty_cache()
            gc.collect()
            if output_queue is not None:
                output_queue.put((idx, train_acc.cpu().numpy().astype(float).tolist(), val_acc.cpu().numpy().astype(float).tolist(), epoch_time)) 
            else:
                return train_acc.cpu().numpy().astype(float).tolist(), val_acc.cpu().numpy().astype(float).tolist(), epoch_time
            
            
    def _determine_train_transform(self):
        print(self.metadata)
        data_augmentations = [
                    v2.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                    v2.RandomVerticalFlip(),    # Randomly flip the image vertically
                    v2.RandomRotation(30),      # Randomly rotate the image by up to 30 degrees
                    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
                    v2.RandomCrop((self.metadata['input_shape'][2],self.metadata['input_shape'][3]), padding=4),  # Randomly crop and resize the image
                    v2.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 0.1
                    v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # Random affine transformation
                    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur with random parameters
                    v2.RandomPerspective(distortion_scale=0.5, p=0.5),  # Apply random perspective transformation
                    v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Randomly erase a rectangle region in the image
        ]
        C,H,W=self.metadata['input_shape'][1:4]
        PH,PW=int(H/8),int(W/8)
        if C==3:
            augmentation_combinations = [
                [],  # No augmentation
                [v2.RandomGrayscale(p=0.2),v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))],
                [v2.RandomGrayscale(p=0.1),v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)), 
                 v2.RandomCrop((H,W), padding=(PH,PW)),
                v2.RandomHorizontalFlip()],
                [ v2.RandomCrop((H,W), padding=(PH,PW)),
                v2.RandomHorizontalFlip()],

                [data_augmentations[7]], # Gaussian blur
     # Grayscale, to tensor
                [data_augmentations[0], data_augmentations[4], data_augmentations[3]],  # Horizontal flip, crop and resize, color jitter
                [data_augmentations[6], data_augmentations[5], data_augmentations[4]],  # Affine transformation, grayscale, crop and resize
                [data_augmentations[0], data_augmentations[4], data_augmentations[7], data_augmentations[9]],  # Horizontal flip, crop and resize, Gaussian blur, random erasing
                #random.sample(data_augmentations, 2),
                #random.sample(data_augmentations, 2),
                #random.sample(data_augmentations, 2),
                #random.sample(data_augmentations, 3),
            ]
        else:
            
            augmentation_combinations= [
                [],
                [ AddNoise(probability=0.1, mean=0.0, stddev=0.1)],
                [v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5))],
                [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW)),
                 v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()],       
           
                [v2.RandomCrop((H,W), padding=(PH,PW)),
                 v2.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomCrop((H,W), padding=(PH,PW)),
                 v2.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                    v2.RandomVerticalFlip(), v2.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))]
            ]

        results = {}
        results_val_acc={}
        

        if not self.multiprocessing:
            for idx, transform in enumerate(augmentation_combinations):
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
                model.to('cuda')
                train_acc, val_acc, epoch_time=self._train_model(model, transform, idx)
                results[str(idx)]={"val_acc":val_acc,
                                   "train_acc":train_acc, 
                                   "epoch_time":epoch_time}
                results_val_acc[str(idx)]=val_acc
        else:
                idx = 0
                output_queue = Queue()
                ic("initial memory")
                print(f"Gpu free memory: {get_gpu_memory(0) / (1024 ** 3):.3f} GB")
                required_memory= 4*2 ** 30
                processes = []
                total_processes_to_run=len(augmentation_combinations)
                while idx < total_processes_to_run:#or any(p.is_alive() for p in processes):
                    if idx<5:
                        sleep_time=3
                    else:
                        sleep_time=10

                    available_memory = get_gpu_memory(0)

                    if (idx < total_processes_to_run) and available_memory>required_memory:
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
                        model.to('cuda')
                        p = mp.Process(target=self._train_model, args=(model, augmentation_combinations[idx], idx, output_queue))
                        p.start()
                        processes.append(p)
                        idx += 1
                        print(f"Gpu free memory: {available_memory / (1024 ** 3):.3f} GB")
                        ic(idx)

                    time.sleep(sleep_time)  # Sleep for a while before checking again


                get_gpu_memory(0)
                for p in processes:
                    p.join()
                
                while not output_queue.empty():
                    idx,train_acc, val_acc, epoch_time=output_queue.get()
               
                    
                    results[str(idx)]={"val_acc":val_acc,
                       "train_acc":train_acc, 
                       "epoch_time":epoch_time}
                    results_val_acc[str(idx)]=val_acc
                    print(results)
                    print(results_val_acc)
                    
        # save the results to a file
        with open(f"tests/augmentations_test/{self.metadata['codename']}augmentation_results.json", 'w') as f:
            json.dump(results, f)

        # return the best transform
        max_key = max(results_val_acc, key=results_val_acc.get)
        max_value = results_val_acc[max_key]

        print(f'The key with the maximum value is "{max_key}" with a value of {max_value}.')
        
        #best_transform = max(results, key=lambda x: x[1])[0]
        return augmentation_combinations[int(max_key)]
