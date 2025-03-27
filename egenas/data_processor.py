import torch
from torchvision.transforms import v2
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
from tests.zcost_proxies.predictive import find_measures
import numpy as np
from trainer import TrainerDistillation, Trainer
from torch.utils.data import DataLoader, TensorDataset
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
from coolname import generate_slug
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

        if len(self.x.shape) == 3:  # Case: [batch_size, height, width]
            # Add a channel dimension to make it [batch_size, 1, height, width]
            self.x = self.x.unsqueeze(1)
        elif len(self.x.shape) == 2:  # Case: [batch_size, feature_dim] or [height, width]
            # If it’s a 2D tensor (e.g., flattened), treat it as [batch_size, feature_dim]
            # Reshape to [batch_size, 1, feature_dim, 1]
            self.x = self.x.unsqueeze(1).unsqueeze(3)
        elif len(self.x.shape) == 4:  # Case: [batch_size, channels, height, width]
            # Already in the correct 4D format, nothing needs to be done
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {self.x.shape}")
        ic(self.x.shape)
        # example transform
        if train:

            self.mean = torch.mean(self.x, [0, 2, 3])
            self.std = torch.std(self.x, [0, 2, 3])
            self.transform_normalization=[transforms.Normalize(self.mean, self.std)]
            self.transform = transforms.Compose(transform+self.transform_normalization)

        else:
            self.transform=v2.Compose(transform)
            #self.transform=None
            
    

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

class RandAugmentMultiChannel(v2.RandAugment):
    def forward(self, img):
        #ic(img.shape)  # Debugging: Check shape before processing

        # Apply RandAugment to each channel separately (ensuring H, W shape for each img[i])
        transformed_channels = [
            v2.RandAugment(self.num_ops, self.magnitude).forward(
                img[i].unsqueeze(0) if img[i].ndim == 2 else img[i]
            ).squeeze(0)  # Remove extra channel dim if added
            for i in range(img.shape[0])
        ]
        #len(transformed_channels)
        return torch.stack(transformed_channels)  # Stack back to (C, H, W)



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
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, select_augment,seed, test_y=None):
        if (seed is not None) and (seed!="None"):
            random_seed = int(seed)
            random.seed(random_seed)
            # Set seed for NumPy
            np.random.seed(random_seed)
            # Set seed for PyTorch
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            # Additional steps if using CuDNN (optional, for GPU acceleration)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.seed=seed
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y=test_y
        self.metadata = metadata
        self.results_path=""
        self.metadata['train_config_path']=f"{self.results_path}configs/train/augmentations_adam.yaml"
        self.select_augment=select_augment
        self.SAVE_PATH=f"{self.results_path}/tests/results/full_training_evonas"
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
        ic(self.select_augment)
        if self.select_augment=="None":
            train_transform=[]
        elif self.select_augment=="Basic":
            C,H,W=self.metadata['input_shape'][1:4]
            PH,PW=int(H/8),int(W/8)
            unique_values=np.unique(self.train_x)
            train_transform=[v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW)),v2.RandomHorizontalFlip()]
        elif self.select_augment=="Proxy":
            train_transform = self._find_train_transform_proxy()
        elif self.select_augment=="Resnet":
            train_transform = self._find_train_transform_resnet()
        elif self.select_augment=="AutoAugment":
            C,H,W=self.metadata['input_shape'][1:4]
            PH,PW=int(H/8),int(W/8)
            unique_values=np.unique(self.train_x)
            train_transform = [v2.AutoAugment()] if C in [1,3] else [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW)),v2.RandomHorizontalFlip()]

        ic(f"selected transform {train_transform}")
        
        
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
            ic("data loaded")

            # get trainer and train
            
            self.metadata["experiment_name"]=f"augmentations_test/{self.metadata['codename']}/aug_{idx}"
            trainer = TrainerDistillation(model, device, train_loader, valid_loader, self.metadata)
            #trainer = Trainer(model, device, train_loader, valid_loader, self.metadata)


            train_acc, val_acc, epoch_time=trainer.train(return_acc=True)        



            print(train_acc.cpu().numpy())
            print(val_acc.cpu().numpy())
            
            del trainer, train_loader, valid_loader, train_ds, valid_ds, model
            torch.cuda.empty_cache()
            gc.collect()

            if output_queue is not None:
                output_queue.put((idx, train_acc.cpu().numpy().astype(float).tolist(), val_acc.cpu().numpy().astype(float).tolist(), epoch_time)) 
            else:
                return train_acc.cpu().numpy().astype(float).tolist(), val_acc.cpu().numpy().astype(float).tolist(), epoch_time           


    def _find_train_transform_resnet(self):
        print(self.metadata)
        C,H,W=self.metadata['input_shape'][1:4]
        PH,PW=int(H/8),int(W/8)
        unique_values=np.unique(self.train_x)

        augmentation_combinations = [

                [],
                [v2.RandAugment(magnitude=9) if C in [1, 3] else RandAugmentMultiChannel()],
                [v2.RandAugment(magnitude=5)],
                [v2.RandAugment(magnitude=1)],
                [v2.TrivialAugmentWide(num_magnitude_bins=31)],
                [v2.TrivialAugmentWide(num_magnitude_bins=15)],
                [v2.AugMix(severity=3)],
                [v2.AugMix(severity=1)],
                ##########################
                [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3)), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)), v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW)), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW)),v2.RandomHorizontalFlip()],
                ###########################################################
                [RandomPixelChange(0.01), v2.ToTensor()],
                [RandomPixelChange(0.025), v2.ToTensor()],
                [RandomPixelChange(0.05), v2.ToTensor()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [RandomPixelChange(0.01), v2.ToTensor(),v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomCrop((H,W), padding=(PH,PW))],
                [RandomPixelChange(0.01), v2.ToTensor(),v2.RandomHorizontalFlip(),v2.RandomVerticalFlip(), v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [v2.AutoAugment()],
                [RandAugmentMultiChannel()]
            ]
            

        results = {}
        results_val_acc={}

        if not self.multiprocessing:
            for idx, transform in enumerate(augmentation_combinations):
                #model_aux, _=regnet_space.create_model(params=params_dict[individual],save_folder=None, name=individual, gen=None, config_updates=None)
                ic(f"Transform {idx}")
                ic(transform)
                try:
                    train_acc, val_acc, epoch_time=self._train_model(transform, idx)

                    results[str(idx)]={"val_acc":val_acc,
                                    "train_acc":train_acc, 
                                    "epoch_time":epoch_time}
                    results_val_acc[str(idx)]=val_acc
                except:
                    ic("Aug not allowed")
                    ic(idx)
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
        with open(f"augmentations_test/{self.metadata['codename']}_results_seed{self.seed}.json", 'w') as f:
            json.dump(results, f)

        
        # Sort the dictionary by value in descending order
        sorted_items = sorted(results_val_acc.items(), key=lambda item: item[1], reverse=True)
        print(f"First best key: {sorted_items[0][0]}")
        print(f"Second best key: {sorted_items[1][0]}")
        
        max_key = sorted_items[0][0] if sorted_items[0][0]!="0" else sorted_items[1][0]
        max_value = results_val_acc[max_key]

        print(f'The key with the maximum value is "{max_key}" with a value of {max_value}.')


        return augmentation_combinations[int(max_key)]
    
    def _find_train_transform_proxy(self):

        regnet_space=RegNet(self.metadata,
                    W0=[16, 120, 8],
                    WA=[16, 64, 8],
                    WM=[2.05,2.9,0.05],
                    D=[8,22,1], 
                    G=[8,8,8], 
                    base_config=f"{self.results_path}configs/search_space/config.yaml")
       
        train_ds = Dataset(self.train_x, self.train_y, train=True, transform=[])
        batch_size = 128

        # build data loaders
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=batch_size,
                                                   drop_last=True,
                                                   shuffle=False)
        #########
        unique_values=np.unique(train_loader.dataset.x[0])
        ic(unique_values)
        C,H,W=self.metadata['input_shape'][1:4]
        ic(C)
        ic(H)
        PH,PW=int(H/8),int(W/8)
        ic(PH)

     
        poss_augs= [

                [],
                [v2.RandAugment(magnitude=9) if C in [1, 3] else RandAugmentMultiChannel()],
                [v2.RandAugment(magnitude=5)],
                [v2.RandAugment(magnitude=1)],
                [v2.TrivialAugmentWide(num_magnitude_bins=31)],
                [v2.TrivialAugmentWide(num_magnitude_bins=15)],
                [v2.AugMix(severity=3)],
                [v2.AugMix(severity=1)],
                ##########################
                [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3)), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)), v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW))],
                [v2.RandomCrop((H,W), padding=(PH,PW)), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),v2.RandomCrop((H,W), padding=(PH,PW)),v2.RandomHorizontalFlip()],
                ###########################################################
                [RandomPixelChange(0.01), v2.ToTensor()],
                [RandomPixelChange(0.025), v2.ToTensor()],
                [RandomPixelChange(0.05), v2.ToTensor()],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()],
                [RandomPixelChange(0.01), v2.ToTensor(),v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],
                [RandomPixelChange(0.01), v2.ToTensor(), v2.RandomCrop((H,W), padding=(PH,PW))],
                [RandomPixelChange(0.01), v2.ToTensor(),v2.RandomHorizontalFlip(),v2.RandomVerticalFlip(), v2.RandomErasing(p=0.2, scale=(0.05, 0.2), ratio=(0.3, 3.3))],#,
                [v2.AutoAugment()]
            ]

                
        ic(poss_augs)
        
        models, chromosomes= regnet_space.create_random_generation(save_folder=None,gen=None, size=20, config_updates=None)
        individuals=list(chromosomes.keys())
        ic(individuals)
        params_dict={}
        for individual in individuals:
            params=[float(chromosomes[individual]["WA"]),int(chromosomes[individual]["W0"]),float(chromosomes[individual]["WM"]),int(chromosomes[individual]["DEPTH"]), int(chromosomes[individual]["GROUP_W"])]
            ic(params)
            params_dict[individual]=params
        del models
        del chromosomes
        gc.collect()
        ic(individuals)
        ic(params_dict)
    
        tot_dfs=[]
        current_transforms=train_loader.dataset.transform.transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for aug in range(len(poss_augs)):

            train_loader.dataset.transform=v2.Compose(poss_augs[aug]+[current_transforms[-1]])
            ic("#############")
            ic(aug)
            ic(train_loader.dataset.transform.transforms)
            ###############
        
            #if "RandAugment"
            train_loader_iter = iter(train_loader)
            # Number of batches to extract
            num_batches =1

            # Initialize lists to hold the inputs and targets from the first 5 batches
            #for batch in range(num_batches):
            inputs_list = []
            targets_list = []
            try:
                for _ in range(num_batches):
                    inputs, targets = next(train_loader_iter)  # Get the next batch
                    inputs_list.append(inputs)
                    targets_list.append(targets)
            except:
                continue
            # Concatenate the inputs and targets across the batches
            inputs = torch.cat(inputs_list)
            targets = torch.cat(targets_list)

            # Create a new TensorDataset from the selected data
            new_dataset = TensorDataset(inputs, targets)

            # Create a new DataLoader from this new dataset
            new_valid_loader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=False)
            #train_loader.batch_size

            syn_scores={}
            measures=["fisher", "jacob_cov"]

            for individual in individuals:
                            #ind_config=f"{folder}/{dataset}/{individual}/config.yaml"
                            #model_aux, _=rg.load_model(config_file=ind_config)
                            model_aux, _=regnet_space.create_model(params=params_dict[individual],save_folder=None, name=individual, gen=None, config_updates=None)
                            #syn_scores[individual]=find_measures(model_aux.to("cuda"), new_valid_loader, ("random",len(new_valid_loader),self.metadata["num_classes"]), "cuda", F.cross_entropy, measures )

                            
                            model_aux.to(device)

                            syn_scores[individual] = find_measures(
                                model_aux, new_valid_loader, 
                                ("random", len(new_valid_loader), self.metadata["num_classes"]), 
                                device, F.cross_entropy, measures
                            )

                            #syn_scores[individual]["params"]=str(params_dict[individuals])
                            del model_aux
                            gc.collect

        #########################################
            tot_df=pd.DataFrame(syn_scores).T.reset_index().rename(columns={"index":"name"})
            tot_df["aug"]=aug
            #tot_df["batch"]=batch
            tot_df["batch"]=0
            tot_dfs.append(tot_df)
        #tot_dfs=pd.concat(tot_dfs)
        df_rank=pd.concat(tot_dfs)


        for col in ["fisher","jacob_cov"]:
            try:
                df_rank[col]=df_rank[col].astype(float)
            except:
                ic("error in col")
                df_rank[col]=0
        df_rank.to_csv(f"../augmentations_test/ranks_{self.metadata['codename']}_seed{self.seed}.csv")

        #############################################################################################
        df_final_rank_jacob_fisher=pd.DataFrame(df_rank.aug.unique(), columns=["aug"])
        sum_ranks=True
        measures=["fisher", "jacob_cov"]
        for name in df_rank.name.unique()[:]:
            df_rank_name=df_rank[df_rank.name==name]
            scaler = MinMaxScaler()
            df_rank_name[measures] = scaler.fit_transform(df_rank_name[measures])
            #########################################################
            for measure in measures:
                df_rank_name[f'rank_{measure}'] = df_rank_name[f'{measure}'].rank(ascending=True, method='dense')
            df_rank_name["fisher_jacob"]=df_rank_name[f'fisher']+df_rank_name[f'jacob_cov']
            df_rank_name[f'rank_fisher_jacob'] = df_rank_name["fisher_jacob"].rank(ascending=True, method='dense')

            if sum_ranks:
                df_final_rank_jacob_fisher=pd.merge(df_final_rank_jacob_fisher, df_rank_name[["aug", "rank_fisher_jacob"]].rename(columns={"rank_fisher_jacob":f"rank_fisher_jacob_{name}"}), on="aug", how="left")
            else:
                df_final_rank_jacob_fisher=pd.merge(df_final_rank_jacob_fisher, df_rank_name[["aug", "fisher_jacob"]].rename(columns={"fisher_jacob":f"rank_fisher_jacob_{name}"}), on="aug", how="left")

        #########################################################################
        df_final_rank_jacob_fisher=pd.DataFrame(df_final_rank_jacob_fisher.set_index("aug").mean(axis=1).rank(ascending=True, method='dense')).rename(columns={0:"jacob_fisher_rank"})
        df_final_rank_jacob_fisher.to_csv(f"../augmentations_test/augmentations_positions_{self.metadata['codename']}_seed{self.seed}.csv")
        best_aug=int(df_final_rank_jacob_fisher[df_final_rank_jacob_fisher.jacob_fisher_rank==1.0].index[0])



        if best_aug==0:
            ic("best aug is 0. Taking next best")
            best_aug=int(df_final_rank_jacob_fisher[df_final_rank_jacob_fisher.jacob_fisher_rank==2.0].index[0])

        ic(f"best_augmentation: {best_aug}")
        return poss_augs[best_aug]