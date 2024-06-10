#from deap import base, creator, tools, algorithms
import numpy as np
from scipy.optimize import dual_annealing
from coolname import generate_slug
from datetime import datetime
import torch
import csv
import os

from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

from trainer import Trainer, TrainerDistillation

from search_space.RegNet import RegNet

class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """
    def __init__(self, train_loader, valid_loader, metadata):
        self.PARAMETERS = {"wa": [8, 48, 8], # start, stop, step
                           "wm": [2.0, 2.9, 0.05], 
                           "depth": [6, 20, 1], 
                           "g": [8, 16, 8],
                           "w0": [8, 56, 8]}
        
        self.regnet_space = RegNet(metadata,
                                   W0=self.PARAMETERS["w0"],
                                   WA=self.PARAMETERS["wa"],
                                   WM=self.PARAMETERS["wm"],
                                   D=self.PARAMETERS["depth"],
                                   G=self.PARAMETERS["g"],
                                   base_config="configs/search_space/config.yaml")

        self.skopt_search_space = self.create_skopt_space()
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata

        self.test_folder = "skopt_" + self.metadata["codename"] + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.mkdir(self.test_folder)
        with open(self.test_folder + '/optimization_history.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameters', 'Performance'])

        self.current_iter = 0
    
    def create_skopt_space(self):
        space = []
        for param, (start, stop, step) in self.PARAMETERS.items():
            if isinstance(start, int) and isinstance(step, int):
                values = list(map(float, range(start, stop + step, step)))
            else:
                values = list(np.around(np.arange(start, stop + step, step, dtype='float32'), decimals=2))
            space.append(Categorical(values, name=param))
        return space


    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """
    def search(self):

        print("Starting NAS...")
        
        best_params = self._optimize()
        model, model_name = self._generate_model(best_params)

        print("Best Model: ", model_name, "With Parameters: ", best_params)

        return model


    def _generate_model(self, model_params):

        # Translte the model_params to the RegNet model
        model_name = str(self.current_iter) + "_" + generate_slug(2).replace("-", "_")
        self.current_iter += 1

        print(model_params)

        model, info = self.regnet_space.create_model(params=[int(model_params[0]), int(model_params[4]), float(model_params[1]), int(model_params[2]), int(model_params[3])],
                                                    save_folder=self.test_folder, name=model_name, gen=None, config_updates=None)

        return model, model_name


    def _evaluate(self, params):
        
        model, model_name = self._generate_model(params)

        self.metadata["train_config_path"]="configs/train/vanilla_generation.yaml"
        self.metadata["experiment_name"]=f"{self.test_folder}/{model_name}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer=TrainerDistillation(model, device, self.train_loader, self.valid_loader, self.metadata) 
        trainer.train()
        
        performance = trainer.best_acc
        performance = performance.cpu().numpy()
        performance = performance / 100

        # save optimization history
        with open(self.test_folder + '/optimization_history.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([params, performance])

        # return negative performance because we want to maximize the performance
        return -performance
        
    
    def _optimize(self):
        result = gp_minimize(self._evaluate, self.skopt_search_space, n_calls=10, n_random_starts=5, n_jobs=-1)

        print("Best Hyperparameters:", result.x)
        print("Best Performance:", result.fun)

        return result.x