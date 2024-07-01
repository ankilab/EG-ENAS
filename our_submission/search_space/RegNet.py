import os
from .config import cfg, load_cfg, reset_cfg
from search_space.models import regnet
from coolname import generate_slug
import torch
import random
import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from .utils import compute_model_size, load_checkpoint
from itertools import product, combinations
from joblib import dump, load
import pandas as pd

class RegNet:
    """
    Class for generating RegNet search space and their configurations.

    Attributes:
        WA_OPTIONS (numpy.ndarray): Array containing the options for the WA (width factor) parameter.
        W0_OPTIONS (numpy.ndarray): Array containing the options for the W0 (initial width) parameter.
        WM_OPTIONS (numpy.ndarray): Array containing the options for the WM (slope) parameter.
        D_OPTIONS (numpy.ndarray): Array containing the options for the D (depth) parameter.
        device (torch.device): The device where the model will be loaded, either 'cuda' if CUDA is available,
            otherwise 'cpu'.

    Example:
        >>> regnet = RegNet(metadata)
    """
    def __init__(self, metadata, W0=[16, 64, 8], WA=[8, 48, 8],WM=[2.05,2.9,0.05],D=[8,22,1],G=[8,16,8], base_config="../configs/search_space/config.yaml"):
        """
        Initializes the RegNet class with default or specified parameters. By default we use a reduced version of the RegNet for testing, but W0, WA and D are usually bigger depending on the dataset specified in metadata.

        Args:
            metadata: Metadata for RegNet models.
            W0 (list): A list containing the initial width parameter options: [start, end, step].
            WA (list): A list containing the width factor parameter options: [start, end, step].
            WM (list): A list containing the slope parameter options: [start, end, step].
            D (list): A list containing the depth parameter options: [start, end, step].
            G (list): A list containing the group width parameter options: [start, end, step].
        """
        reset_cfg()
        cfg.merge_from_file(base_config)
        cfg.MODEL.NUM_CLASSES=metadata["num_classes"]
        cfg.REGNET.STEM_W=metadata["input_shape"][-1]
        cfg.REGNET.INPUT_CHANNELS=metadata["input_shape"][1]
        self.cfg=cfg
        
        self.WA_STEP=WA[2]
        self.W0_STEP=W0[2]
        self.WM_STEP=WM[2]
        self.D_STEP=D[2]
        self.WA_OPTIONS=np.arange(WA[0],WA[1]+WA[2], WA[2])
        self.W0_OPTIONS=np.arange(W0[0],W0[1]+W0[2], W0[2])
        self.WM_OPTIONS=np.arange(WM[0],WM[1]+WM[2], WM[2])
        self.D_OPTIONS=np.arange(D[0],D[1]+D[2],D[2])
        self.G_OPTIONS=np.arange(G[0],G[1]+G[2],G[2])
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_model(self, params=None, save_folder=None, name=None, gen=None, config_updates=None):
        """
        Constructs a model either randomly or based on the specified parameters (WA,W0,WM,D).

        Args:
            params (list, optional): A list containing specific parameters (WA,W0,WM,D) for configuring the model. If None the values are selected randomly.
            save_folder (str, optional): Study folder to save the new models. If None, the config file of the model is not saved.
            name (str, optional): Name of the model. If None, a randomly generated name for the model is used.
            gen (int, optional): The generation number associated with the model. Used for saving the model in the generation folder during NAS. If None, the model is directly saved in save folder.

        Returns:
            tuple: A tuple containing the constructed model and information about its configuration.

        Example (create random model from RegNet):
            >>> model, info = regnet.create_model( save_config="test_1")
            >>> print(info)
            {'ws': [40, 104],
             'bs': [1.0, 1.0],
             'gs': [8, 8],
             'ds': [4, 4],
             'num_stages': 2,
             'total_size_mb': None,
             'h': 1,
             'w': 1,
             'flops': 32546,
             'params': 176482,
             'acts': 802,
             'WA': 8.0,
             'W0': 40,
             'WM': 2.65,
             'DEPTH': 8}

        """
        # Load the config

        if params is None:
            cfg.REGNET.WA=float(random.choice(self.WA_OPTIONS))
            cfg.REGNET.W0=int(random.choice(self.W0_OPTIONS))
            cfg.REGNET.WM=float(random.choice(self.WM_OPTIONS))
            cfg.REGNET.DEPTH=int(random.choice(self.D_OPTIONS))
            cfg.REGNET.GROUP_W=int(random.choice(self.G_OPTIONS))
        else:
            cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH, cfg.REGNET.GROUP_W = params
            #cfg.REGNET.GROUP_W=8
        
        if cfg.REGNET.WA>cfg.REGNET.W0:
            cfg.REGNET.W0=int(random.choice([option for option in self.W0_OPTIONS if option >= cfg.REGNET.WA]))
            #print("Corrected W0: ", cfg.REGNET.W0)
        
        _, _, num_stages, _,_,_=self._generate_regnet(cfg.REGNET.WA,cfg.REGNET.W0,cfg.REGNET.WM,cfg.REGNET.DEPTH, q=8)         
        i=0
        while num_stages>5:
            #print("Num stages: ", num_stages)
            #print("WM: ",cfg.REGNET.WM)
            #print("DEPTH: ",cfg.REGNET.DEPTH)
            #print("WA: ",cfg.REGNET.WA)
            #print("W0: ",cfg.REGNET.W0)
            cfg.REGNET.WM=min(cfg.REGNET.WM+0.1,max(self.WM_OPTIONS))
            cfg.REGNET.DEPTH=max(cfg.REGNET.DEPTH-2,min(self.D_OPTIONS))
            if i==3:
              cfg.REGNET.WA=max(cfg.REGNET.WA-self.WA_STEP,min(self.WA_OPTIONS))
              cfg.REGNET.W0=min(cfg.REGNET.W0+self.W0_STEP,max(self.W0_OPTIONS))
            _, _, num_stages, _,_,_=self._generate_regnet(cfg.REGNET.WA,cfg.REGNET.W0,cfg.REGNET.WM,cfg.REGNET.DEPTH, q=8) 
            
            i=i+1
    
        
        if config_updates is not None:
            cfg.merge_from_list(config_updates)
            
        # Write the dictionary to a YAML file
        if save_folder is not None:
            if name is None:
                name = generate_slug(2).replace("-", "_")
            print("Created model: ", name)

            #config_folder=os.path.dirname(config_file)
            if gen is not None:
                output_directory=f"{save_folder}/Generation_{gen}/{name}"
            else:
                output_directory=f"{save_folder}/{name}"

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            output_file_path = f"{output_directory}/config.yaml"
            with open(output_file_path, "w") as f:
              f.write(cfg.dump()) 

        # Construct model
        model=regnet.RegNet().to(self.device)
        # Load pretrained weights
        info=self._get_blocks_per_stage()
        total_size_mb=compute_model_size(model)
        info["total_size_mb"]=total_size_mb
        info.update(model.complexity({"h":0, "w":0, "flops":0, "params":0, "acts":0}))
        info["WA"]=cfg.REGNET.WA
        info["W0"]=cfg.REGNET.W0
        info["WM"]=cfg.REGNET.WM
        info["DEPTH"]=cfg.REGNET.DEPTH
        info["GROUP_W"]=cfg.REGNET.GROUP_W
        return model, info

    def load_model(self,config_file, weights_file=None,  config_updates=None):
        """
        Constructs a predefined model based on the specified configuration file and optionally loads pretrained weights.

        Args:
            config_file (str): The file path to the configuration file.
            weights_file (str, optional): The file path to the pretrained weights file.

        Returns:
            tuple: A tuple containing the constructed model and information about its configuration.

        Example:
            >>> model, info = regnet.load_model('config.yaml', weights_file='model_weights.pth')

        """
                
        # Load the config
        assert os.path.exists(config_file), f"Configuration file '{config_file}' does not exist."
        
        reset_cfg()
        cfg.merge_from_file(config_file)

        if config_updates is not None:
            cfg.merge_from_list(config_updates)

        # Construct model
        model=regnet.RegNet().to(self.device)
        print("Loading model:", config_file)
        # Load pretrained weights
        if weights_file is not None:
            state = load_checkpoint(weights_file)
            model.load_state_dict(state["model"])

        info=self._get_blocks_per_stage()
        total_size_mb=compute_model_size(model)
        #info["total_params"]=total_params
        info["total_size_mb"]=total_size_mb
        info.update(model.complexity({"h":0, "w":0, "flops":0, "params":0, "acts":0}))
        info["WA"]=cfg.REGNET.WA
        info["W0"]=cfg.REGNET.W0
        info["WM"]=cfg.REGNET.WM
        info["DEPTH"]=cfg.REGNET.DEPTH
        info["GROUP_W"]=cfg.REGNET.GROUP_W
        return model, info

    # Create a new column by applying the function to each row
    def get_ranking(self, ranking_test_df, test_column):
        ranking_predict={}
        for ind in list(ranking_test_df.name_A.unique())+list(ranking_test_df.name_B.unique()):
            ranking_predict[ind]=0
        for index, row in ranking_test_df.iterrows():
            if row[test_column]==1:
                ranking_predict[row["name_A"]]=ranking_predict[row["name_A"]]+1
            else:
                ranking_predict[row["name_B"]]=ranking_predict[row["name_B"]]+1
        ranking_predict_df=pd.DataFrame([ranking_predict]).T.rename(columns={0:"score"}).sort_values(by="score", ascending=False)
        return ranking_predict_df
    
    def create_first_generation(self, save_folder,gen, size, config_updates=None, metadata=None):
        # Create the Cartesian product of these values
        models, chromosomes=self.create_random_generation(save_folder=None,gen=None, size=size*5, config_updates=None)
        rf_classifier=load(f'tests/classifiers/{metadata["codename"]}/rfc_model.joblib')
        
        gen_df=pd.DataFrame(chromosomes).T.reset_index().rename(columns={"index":"name"})[["name","num_stages","params","WA","W0","WM","DEPTH"]]
        
        pairs = list(combinations(gen_df.index, 2))
        combined_data = []

        for idx1, idx2 in pairs:
            row1 = gen_df.loc[idx1]
            row2 = gen_df.loc[idx2]
            combined_row = {
                'name_A': row1['name'],
                'name_B': row2['name'],
                'num_stages_A': row1['num_stages'],
                'params_A': row1['params'],
                'WA_A': row1['WA'],
                'W0_A': row1['W0'],
                'WM_A': row1['WM'],
                'DEPTH_A': row1['DEPTH'],
                'num_stages_B': row2['num_stages'],
                'params_B': row2['params'],
                'WA_B': row2['WA'],
                'W0_B': row2['W0'],
                'WM_B': row2['WM'],
                'DEPTH_B': row2['DEPTH'],
                #'label': 1 if row1['best_acc'] > row2['best_acc'] else 0
            }

            combined_data.append(combined_row)

        combined_df = pd.DataFrame(combined_data)
        combined_df["benchmark"]=metadata["benchmark"]
        combined_df["num_classes"]=metadata["num_classes"]
        combined_df["num_channels"]=metadata["input_shape"][1]

        cols_train=[ 
        'num_stages_A', 'WA_A', 'W0_A', "params_A",
       'WM_A', 'DEPTH_A',
        'num_stages_B', 'WA_B', 'W0_B', "params_B",
       'WM_B', 'DEPTH_B',"num_classes", "benchmark", "num_channels"]
        X_test=combined_df[cols_train]
        
        ranking_test_df=combined_df[["name_A","name_B"]]
        pred_column="rf_prediction"
        ranking_test_df[pred_column]=rf_classifier.predict(X_test)
        
        ranking_prediction_df=self.get_ranking(ranking_test_df, pred_column).head(size)
        print(ranking_prediction_df)
        best_individuals=list(ranking_prediction_df.index)

        best_models={}
        best_chromosomes={}
        for ind in best_individuals:
            chrom=chromosomes[ind]
            wa,w0,wm,d,group_w=chrom["WA"],chrom["W0"],chrom["WM"],chrom["DEPTH"], chrom["GROUP_W"]
            best_models[ind], best_chromosomes[ind]=self.create_model(params=[float(wa),int(w0),float(wm),int(d), int(group_w)],
                                      save_folder=save_folder, 
                                      name=ind, 
                                      gen=gen, 
                                      config_updates=config_updates)
            
        return best_models, best_chromosomes
    
    def create_generation(self,params, save_folder,gen, config_updates=None):
        models={}
        chromosomes={}
        for name,param in params.items():
            wa,w0,wm,d=param
            group_w=int(random.choice(self.G_OPTIONS))
            #random_name = generate_slug(2).replace("-", "_")
            model, info=self.create_model(params=[float(wa),int(w0),float(wm),int(d), int(group_w)],save_folder=save_folder, name=name, gen=gen, config_updates=config_updates)
            models[name]=model
            chromosomes[name]=info
            
        return models, chromosomes
    
    
    def create_random_generation(self, save_folder,gen, size, config_updates=None):
        """
        Creates a random generation of models with specified size and saves them to the specified folder.

        Args:
            save_folder (str): The folder path where the generated models will be saved.
            gen (int): The generation number associated with the models.
            size (int): The number of models to generate.

        Returns:
            tuple: A tuple containing dictionaries of generated models and their corresponding chromosome information.
        """

        models={}
        chromosomes={}
        for ind in range(size):
            random_name = generate_slug(2).replace("-", "_")
            model, info=self.create_model(save_folder=save_folder, name=random_name, gen=gen, config_updates=config_updates)
            models[random_name]=model
            chromosomes[random_name]=info
        return models, chromosomes

    def load_generation(self,folder, config_updates=None):
        """
        Loads a generation of models from the specified folder.

        Args:
            folder (str): The folder path containing the models to load.

        Returns:
            tuple: A tuple containing dictionaries of loaded models and their corresponding chromosome information.
        """
        
        models={}
        chromosomes={}
        #configs={}
        individuals=os.listdir(folder)
        individuals=[ind for ind in individuals if os.path.isdir(os.path.join(folder, ind)) and ".ipynb" not in ind]
        for ind in individuals:
            ind_config=f"{folder}/{ind}/config.yaml"
            models[ind], chromosomes[ind]=self.load_model(config_file=ind_config, config_updates=config_updates)
        return models,chromosomes

    def compare_chromosomes(self, c1, c2):
        #wa,w0,wm,D
        max_range=[max(self.WA_OPTIONS), max(self.W0_OPTIONS), max(self.WM_OPTIONS), max(self.D_OPTIONS)]
        min_range=[min(self.WA_OPTIONS), min(self.W0_OPTIONS),min(self.WM_OPTIONS),min(self.D_OPTIONS)]
        ranges=np.array(max_range)-np.array(min_range)
        diff=np.abs(np.array(c1)-np.array(c2))/ranges
        diff[2]=np.sqrt(diff[2])
        return diff.mean()
    
    def _adjust_block_compatibility(self, ws, bs, gs):
        """Adjusts the compatibility of widths, bottlenecks, and groups."""
        assert len(ws) == len(bs) == len(gs)
        assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
        assert all(b < 1 or b % 1 == 0 for b in bs)
        vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
        gs = [int(min(g, v)) for g, v in zip(gs, vs)]
        ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
        vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
        ws = [int(v / b) for v, b in zip(vs, bs)]
        assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
        return ws, bs, gs

    def _generate_regnet(self,w_a, w_0, w_m, d, q=8):
        """Generates per stage widths and depths from RegNet parameters."""
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
        # Generate continuous per-block ws
        ws_cont = np.arange(d) * w_a + w_0
        #print("ws_cont: ",ws_cont)
        # Generate quantized per-block ws
        ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
        #print("ks: ",ks)
        ws_all = w_0 * np.power(w_m, ks)
        ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
        #print("ws_all:", ws_all)
        # Generate per stage ws and ds (assumes ws_all are sorted)
        ws, ds = np.unique(ws_all, return_counts=True)
        # Compute number of actual stages and total possible stages
        num_stages, total_stages = len(ws), ks.max() + 1
        # Convert numpy arrays to lists and return
        ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
        return ws, ds, num_stages, total_stages, ws_all, ws_cont

    def _get_blocks_per_stage(self):
            ws, ds, num_stages, total_stages, ws_all, ws_cont=self._generate_regnet(w_a=cfg.REGNET.WA, w_0=cfg.REGNET.W0,    w_m=cfg.REGNET.WM, d=cfg.REGNET.DEPTH)
            ss = [cfg.REGNET.STRIDE for _ in ws]
            bs = [cfg.REGNET.BOT_MUL for _ in ws]
            gs = [cfg.REGNET.GROUP_W for _ in ws]
            ws, bs, gs = self._adjust_block_compatibility(ws, bs, gs)
            info={"ws":ws,"bs":bs,"gs":gs,"ds":ds,"num_stages":num_stages}
            return info
