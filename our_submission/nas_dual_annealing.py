#from deap import base, creator, tools, algorithms
import numpy as np
from scipy.optimize import dual_annealing
from coolname import generate_slug
from datetime import datetime
import torch
import csv

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
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata

        self.test_folder = "test_dual_annealing-" + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.current_iter = 0
        self.optimization_history = []


    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """
    def search(self):
        
        bounds = [(0, 1) for _ in range(len(self.PARAMETERS))]

        result = dual_annealing(self._objective_function, bounds, maxiter=15, callback=self.callback)

        best_params = result.x
        model = self._generate_model(best_params)

        # save optimization history
        with open(self.test_folder + '/optimization_history.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameters', 'Function Value'])
            for entry in optimization_history:
                writer.writerow([entry[0], entry[1]])

        return model

    def _callback(x, f, context):
        self.optimization_history.append((x, f))

    def _generate_model(self, params):
        
        # convert values between [0, 1] to actual values defined in self.PARAMETERS
        model_params = {}
        for i, key in enumerate(self.PARAMETERS.keys()):
            start, stop, step = self.PARAMETERS[key]
            value = start + (stop - start) * params[i]
            value = round((value - start) / step) * step + start

            model_params[key] = value

        # Translte the model_params to the RegNet model
        model_name = str(self.current_iter) + "_" + generate_slug(2).replace("-", "_")
        self.current_iter += 1

        model, info = self.regnet_space.create_model(params=[model_params["wa"], model_params["w0"], model_params["wm"], model_params["depth"], model_params["g"]],
                                                    save_folder=self.test_folder, name=model_name, gen=None, config_updates=None)

        return model, model_name

    def _evaluate_model(self, model, model_name):
        self.metadata["train_config_path"]="configs/train/vainilla_generation_complete.yaml"
        self.metadata["experiment_name"]=f"{self.test_folder}/{model_name}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer=TrainerDistillation(model, device, self.train_loader, self.valid_loader, self.metadata) 
        trainer.train()
        
        performance = trainer.best_acc
        performance = performance.cpu().numpy()
        
        return performance
    
    def _objective_function(self, params):
        model, model_name = self._generate_model(params)
        performance = self._evaluate_model(model, model_name)
        return performance