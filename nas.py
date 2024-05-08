#from deap import base, creator, tools, algorithms
import random
import numpy as np

from scipy.optimize import dual_annealing

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
                           "w0": [8, 56, 8]}


    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """
    def search(self):
        
        bounds = [(0, 1) for _ in range(len(self.PARAMETERS))]

        result = dual_annealing(self._objective_function, bounds, maxiter=100)

        best_params = result.x
        model = self._generate_model(best_params)

        return model

    def _generate_model(self, params):
        # convert values between [0, 1] to actual values defined in self.PARAMETERS
        
        model_params = {}
        for i, key in enumerate(self.PARAMETERS.keys()):
            start, stop, step = self.PARAMETERS[key]
            value = start + (stop - start) * params[i]
            value = round((value - start) / step) * step + start

            model_params[key] = value

        print(model_params)

        # TODO: params --> pytorch model

        return None

        return model

    def _evaluate_model(self, model):
        # TODO: model --> training --> performance on test set
        
        return np.random.rand()
        return performance
    
    def _objective_function(self, params):
        model = self._generate_model(params)
        performance = self._evaluate_model(model)
        return performance