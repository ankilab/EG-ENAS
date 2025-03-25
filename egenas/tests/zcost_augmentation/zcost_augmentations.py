import json
import math
import numpy as np
import pickle as pkl
import os
import time

import torch
from torch.utils.data import RandomSampler
import pandas as pd
from data_processor import DataProcessor
from trainer import Trainer
import argparse
import torchvision.models as models
import sys
from torchvision.models.vision_transformer import VisionTransformer
import random
#sys.path.append("egenas/tests/zcost_augmentation")

# === DATA LOADING HELPERS =============================================================================================
# find the dataset filepaths
def get_dataset_paths(data_dir):
    paths = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if 'dataset' in d], reverse=True)
    return paths

# load the dataset metadata from json
def load_dataset_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'metadata'), "r") as f:
        metadata = json.load(f)
    return metadata

# load dataset from file
def load_datasets(data_path, truncate):
    data_path = '/home/woody/iwb3/iwb3021h/datasets/'+data_path
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


# === TIME COUNTERs ====================================================================================================
def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


# keep a counter of available time
class Clock:
    def __init__(self, time_available):
        self.start_time =  time.time()
        self.total_time = time_available

    def check(self):
        return self.total_time + self.start_time - time.time()


# === MODEL ANALYSIS ===================================================================================================
def general_num_params(model):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


if __name__ == '__main__':
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    # === MAIN =============================================================================================================
    # the available runtime will change at various stages of the competition, but feel free to change for local tests
    # note, this is approximate, your runtime will be controlled externally by our server
    total_runtime_hours = 30
    total_runtime_seconds = total_runtime_hours * 60 * 60

    for seed in [6,7,8,9,10]:
        for dataset in ["AddNIST","Chesseract","CIFAR10","CIFARTile","GeoClassing","Gutenberg","ImageNet16-120", "Language","MultNIST","Sudoku","Voxel"]:
            #dataset="Voxel"
            #select_augment="Model"
            select_augment="Proxy"
            #seed=1
            #model_name="EfficientNet_b0"
            model_name="RegNetY_400MF"

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

            runclock = Clock(total_runtime_seconds)


            # load and display data info
            (train_x, train_y), (valid_x, valid_y), (test_x), metadata = load_datasets(dataset, truncate=False)
            metadata['time_remaining'] = runclock.check()
            start_time = time.time()

            print("Metadata:")
            [print("   - {:<20}: {}".format(k, v)) for k,v in metadata.items()]

            # perform data processing/augmentation/etc using your DataProcessor
            print("\n=== Processing Data ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))

            #### Select model
            #model = ###
            metadata["model_name"]=model_name
            if model_name=="ResNet18":
                model = models.resnet18(weights=None)
                new_conv1 = torch.nn.Conv2d(in_channels=metadata["input_shape"][1], 
                                        out_channels=model.conv1.out_channels, 
                                        kernel_size=model.conv1.kernel_size, 
                                        stride=model.conv1.stride, 
                                        padding=model.conv1.padding, 
                                        bias=model.conv1.bias)
                # Replace the first convolutional layer
                model.conv1 = new_conv1
                model.fc = torch.nn.Linear(512, metadata['num_classes'])
                model.to(device)
                ####
            if model_name == "RegNetY_400MF":
                # Load a predefined RegNet architecture
                model = models.regnet_y_400mf(weights=None)  # Using no pre-trained weights

                # Adjust the first convolution layer for the specific input channels
                new_stem_conv = torch.nn.Conv2d(
                    in_channels=metadata["input_shape"][1],  # Dynamic input channels
                    out_channels=model.stem[0].out_channels,  # Preserve original out channels
                    kernel_size=model.stem[0].kernel_size,
                    stride=model.stem[0].stride,
                    padding=model.stem[0].padding,
                    bias=model.stem[0].bias
                )
                # Replace the stem's initial convolutional layer
                model.stem[0] = new_stem_conv

                # Replace the fully connected layer with a dynamic output layer
                model.fc = torch.nn.Linear(model.fc.in_features, metadata['num_classes'])

                # Move the model to the specified device
                model.to(device)

            if model_name == "VGG19":
                model = models.vgg19(weights=None)
                # Modify the first convolutional layer
                model.features[0] = torch.nn.Conv2d(
                    in_channels=metadata["input_shape"][1],
                    out_channels=model.features[0].out_channels,
                    kernel_size=model.features[0].kernel_size,
                    stride=model.features[0].stride,
                    padding=model.features[0].padding
                )
                
                # Add a global average pooling before the classifier
                model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

                # Modify the classification layer
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, metadata['num_classes'])

                # Move model to device
                model.to(device)


            if model_name == "MobileNetV3_large":
                model = models.mobilenet_v3_large(weights=None)
                # Modify the first convolutional layer
                model.features[0][0] = torch.nn.Conv2d(
                    in_channels=metadata["input_shape"][1],
                    out_channels=model.features[0][0].out_channels,
                    kernel_size=model.features[0][0].kernel_size,
                    stride=model.features[0][0].stride,
                    padding=model.features[0][0].padding,
                    bias=model.features[0][0].bias
                )
                # Modify the classification layer
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, metadata['num_classes'])
                model.to(device)


            if model_name == "EfficientNet_b0":
                model = models.efficientnet_b0(weights=None)
                # Modify the first convolutional layer
                model.features[0][0] = torch.nn.Conv2d(
                    in_channels=metadata["input_shape"][1],
                    out_channels=model.features[0][0].out_channels,
                    kernel_size=model.features[0][0].kernel_size,
                    stride=model.features[0][0].stride,
                    padding=model.features[0][0].padding,
                    bias=model.features[0][0].bias
                )
                # Modify the classification layer
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, metadata['num_classes'])
                model.to(device)

            ####

            data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, select_augment, model, seed)
            train_loader, valid_loader, test_loader = data_processor.process()
            metadata['time_remaining'] = runclock.check()

            # check that the test_loader is configured correctly
            
            #assert_string = "Test Dataloader is {}, this will break evaluation. Please fix this in your DataProcessor init."
            #assert not isinstance(test_loader.sampler, RandomSampler), assert_string.format("shuffling")
            #assert not test_loader.drop_last, assert_string.format("dropping last batch")

            # search for best model using your NAS algorithm
            print("\n=== Performing NAS ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))



            #model_params = int(general_num_params(model))
            #metadata['time_remaining'] = runclock.check()

            # train model using your Trainer
            #print("\n=== Training ===")
            #print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            #device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
            #trainer = Trainer(model, device, train_loader, valid_loader, metadata)
            #trained_model = trainer.train()

            # submit predictions to file
            #print("\n=== Predicting ===")
            #print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            #predictions = trainer.predict(test_loader)
            #run_data = {'Runtime': float(np.round(time.time()-start_time, 2)), 'Params': model_params}
            #with open("predictions/{}_stats.pkl".format(metadata['codename']), "wb") as f:
            #    pkl.dump(run_data, f)
            #np.save('predictions/{}.npy'.format(metadata['codename']), predictions)



            #Adaline, Caitie, Chester, CIFAR10, Gutenberg, in16, LaMelo, Mateo, Sokoto, Volga
            #MobileNetV3_large, EfficientNet_b0,  RegNetY_400MF

            ################################################################################
            #model_name="EfficientNet_b0"
            #for dataset in ["Adaline", "Caitie", "Chester", "CIFAR10", "Gutenberg", "in16", "LaMelo", "Mateo", "Sokoto", "Volga", "Sadie"]:
            def results_to_df(path):
                data = []
                # Open the text file
                with open(path, 'r') as file:
                    lines = file.readlines()
                    # Initialize an empty dictionary to store data for each block
                    block_data = {}
                    for line in lines:
                        # If the line contains dashes, it indicates the end of a block
                        if '-------------------------' in line:
                            # If block_data is not empty, add it to the list of data dictionaries
                            if block_data:
                                data.append(block_data)
                                # Reset block_data for the next block
                                block_data = {}
                        elif 'best_acc' in line:
                            continue
                        else:
                            # Split the line by ':'
                            #print(line)
                            key, value = line.strip().split(': ')
                            # Store the key-value pair in the block_data dictionary
                            block_data[key] = value

                # Create a DataFrame from the list of dictionaries
                df = pd.DataFrame(data)

                # Convert columns to appropriate data types if needed
                df['epoch'] = df['epoch'].astype(int)
                df['lr'] = df['lr'].astype(float)
                df['train_acc'] = df['train_acc'].astype(float)
                df['train_loss'] = df['train_loss'].astype(float)
                df['test_acc'] = df['test_acc'].astype(float)
                df['test_acc_top5'] = df['test_acc_top5'].astype(float)
                df['test_loss'] = df['test_loss'].astype(float)
                df['epoch_time'] = df['epoch_time'].astype(float)

                
                return df


            if select_augment=="Proxy":
                dataset=metadata["codename"]
                rows=[]
                for aug in range(23):
                    try:
                        df_scores=results_to_df(f"/home/woody/iwb3/iwb3021h/augmentations_test/{model_name}/{dataset}/aug_{aug}/worklog.txt")

                        row_max = df_scores.loc[[df_scores['test_acc'].idxmax()]]
                        row_max=row_max.assign(aug=aug)
                        rows.append(row_max[["train_acc","test_acc","test_loss", "aug"]])
                    except:
                        print(aug)        
                total_scores_df=pd.concat(rows)

                #################
                measures=["epe_nas","nwot","plain","snip", "fisher","jacob_cov", "grad_norm","grasp"]
                zcosts=pd.read_csv(f"/home/woody/iwb3/iwb3021h/augmentations_test/{model_name}/{dataset}/ranks_{dataset}_{seed}.csv")#, index_col=0)
                try: 
                    zcosts=zcosts[measures+["aug"]]#"synflow","l2_norm"
                except:
                    measures=["epe_nas","nwot","plain","snip", "fisher","jacob_cov", "grad_norm"]
                    zcosts=zcosts[measures+["aug"]]#"synflow","l2_norm"
                ####################

                zcosts=pd.merge(total_scores_df, zcosts, on="aug", how="left").sort_values(by="test_acc", ascending=False)

                # Check for columns with complex values
                for col in zcosts.select_dtypes(include=['O']).columns:
                    # Convert to complex and take only the real part
                    zcosts[col] = pd.to_numeric(zcosts[col], errors='coerce', downcast=None).apply(
                        lambda x: x.real if pd.notnull(x) else x
                    )

                zcosts.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Fill NaN (previously inf) with the column means
                zcosts.fillna(zcosts.mean(), inplace=True)

                #Correlation
                spearman_corr = zcosts.corr(method='spearman')
                spearman_corr.to_csv(f"/home/woody/iwb3/iwb3021h/augmentations_test/{model_name}/{dataset}/corr_{dataset}_{seed}.csv")

                spearman_corr=spearman_corr[["test_acc","aug","test_loss"]].iloc[4:]
                spearman_corr["model"]=model_name
                spearman_corr["dataset"]=dataset
                #correlations.append(spearman_corr)
                #########################
                #Scaling
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                zcosts[measures] = scaler.fit_transform(zcosts[measures])
                #########################################################
                for measure in measures:
                    zcosts[f'rank_{measure}'] = zcosts[f'{measure}'].rank(ascending=False, method='dense')
                zcosts["fisher_jacob"]=zcosts[f'fisher']+zcosts[f'jacob_cov']
                zcosts[f'rank_fisher_jacob'] = zcosts["fisher_jacob"].rank(ascending=False, method='dense')
                #zcosts[f'rank_fisher_jacob'] = (zcosts[f'fisher']+zcosts[f'jacob_cov']).rank(ascending=False, method='dense')
                zcosts["model"]=model_name
                zcosts.to_csv(f"/home/woody/iwb3/iwb3021h/augmentations_test/{model_name}/{dataset}/zcosts_{dataset}_{seed}.csv")

