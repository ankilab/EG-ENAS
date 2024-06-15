import torch
from torchvision.transforms import v2
import torchvision.models as models
import numpy as np
from utils.transforms import get_train_transform
from trainer import Trainer
# import torchvision.transforms as transforms
import json
import random


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
    def process(self):
        # Try different transforms for training, we select the best one and use it
        train_transform = self._determine_train_transform()
        
        # Create dataloaders with final transforms
        train_ds = Dataset(self.train_x, self.train_y, train=True, transform=train_transform)
        valid_ds = Dataset(self.valid_x, self.valid_y, train=False, transform=train_ds.transform_normalization)
        test_ds = Dataset(self.test_x, self.test_y, train=False, transform=train_ds.transform_normalization)

        batch_size = 64

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
    
    def _determine_train_transform(self):
        data_augmentations = [
                    v2.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                    v2.RandomVerticalFlip(),    # Randomly flip the image vertically
                    v2.RandomRotation(30),      # Randomly rotate the image by up to 30 degrees
                    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Randomly change brightness, contrast, saturation, and hue
                    v2.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Randomly crop and resize the image
                    v2.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 0.1
                    v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # Random affine transformation
                    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur with random parameters
                    v2.RandomPerspective(distortion_scale=0.5, p=0.5),  # Apply random perspective transformation
                    v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Randomly erase a rectangle region in the image
        ]

        augmentation_combinations = [
            [],  # No augmentation
            [data_augmentations[7]], # Gaussian blur
            [data_augmentations[5]], # Grayscale
            [data_augmentations[0], data_augmentations[4], data_augmentations[3]],  # Horizontal flip, crop and resize, color jitter
            [data_augmentations[6], data_augmentations[5], data_augmentations[4]],  # Affine transformation, grayscale, crop and resize
            [data_augmentations[0], data_augmentations[4], data_augmentations[7], data_augmentations[9]],  # Horizontal flip, crop and resize, Gaussian blur, random erasing
            random.sample(data_augmentations, 2),
            random.sample(data_augmentations, 2),
            random.sample(data_augmentations, 2),
            random.sample(data_augmentations, 3),
        ]

        results = []

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        for i, transform in enumerate(augmentation_combinations):

            train_ds = Dataset(self.train_x, self.train_y, train=True, transform=transform)
            valid_ds = Dataset(self.valid_x, self.valid_y, train=False, transform=train_ds.transform_normalization)

            try:
                # get ResNet-18 model
                model = models.resnet18(weights=None)
                new_conv1 = torch.nn.Conv2d(in_channels=self.metadata["input_shape"][1], out_channels=model.conv1.out_channels, 
                        kernel_size=model.conv1.kernel_size, 
                        stride=model.conv1.stride, 
                        padding=model.conv1.padding, 
                        bias=model.conv1.bias)
                

                # Replace the first convolutional layer
                model.conv1 = new_conv1
                model.fc = torch.nn.Linear(512, self.metadata['num_classes'])
                model.to('cuda')
            except:
                # Augmentation is not working with the model, so we skip it
                results.append((i, 0))
                continue

            # get dataloaders
            batch_size = 128
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, drop_last=True, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

            # get trainer and train
            trainer = Trainer(model, device, train_loader, valid_loader, self.metadata)
            trainer.epochs = 10
            trainer.train()
            acc = trainer.evaluate()
            
            results.append((i, acc))

        # save the results to a file
        with open('augmentation_results.json', 'w') as f:
            json.dump(results, f)

        # return the best transform
        best_transform = max(results, key=lambda x: x[1])[0]
        return augmentation_combinations[best_transform]

