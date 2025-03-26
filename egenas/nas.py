import os
import copy
import sys
import ast
import torch
import random
import numpy as np
import time
from IPython.display import clear_output
from icecream import ic
import gc
import functools


######## Search space #########
from search_space.RegNet import RegNet
from search_space.utils import create_widths_plot, scatter_results, get_generation_dfs
######## Training ###############
from trainer import Trainer, TrainerDistillation
from utils.train_cfg import get_cfg, show_cfg
###################################################

##################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datetime import datetime
import itertools
import pandas as pd
import json
from io import StringIO
from coolname import generate_slug
from sklearn.metrics import accuracy_score
import torch.multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo



def get_gpu_memory(gpu_id):
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    return info.free

class NAS:
    def __init__(self, train_loader, valid_loader, metadata,mode,select_augment,seed,pretrained_pool_path, resume_from=None, test=False):
        self.test=test
        self.SUBMISSION_PATH=""

        ic(mode)
        ic(seed)
        ic(select_augment)
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
        else:
            random_seed="random"

        epochs=10 if mode=="T7" else 5
        SAVE_PATH=f"{mode}_{select_augment}/seed_{random_seed}"
        #SAVE_PATH=f"results/THESIS_RESULTS/T1"
        if "+" in mode:
            ic("Extended + Regnet mode")
            self.regnet_space=RegNet(metadata,
                        W0=[64, 240, 8],
                        WA=[32, 128, 8],
                        WM=[2.05,2.9,0.05],
                        D=[12,25,1], 
                        G=[8,8,8], 
                        base_config=f"{self.SUBMISSION_PATH}configs/search_space/config.yaml")
        else:
            ic("No extended Regnet")
            self.regnet_space=RegNet(metadata,
            W0=[16, 120, 8],
            WA=[16, 64, 8],
            WM=[2.05,2.9,0.05],
            D=[8,22,1], 
            G=[8,8,8], 
            base_config=f"{self.SUBMISSION_PATH}configs/search_space/config.yaml")

        current_date= datetime.now().strftime("%d_%m_%Y_%H_%M")
       
        
        self.metadata=metadata
        self.metadata["test_type"]=f"{mode}_{select_augment}/seed_{random_seed}"
        if epochs==10:
            self.metadata["train_config_path"]=f"{self.SUBMISSION_PATH}configs/train/T10.yaml" 
        elif epochs==5:
            self.metadata["train_config_path"]=f"{self.SUBMISSION_PATH}configs/train/T5.yaml" 
        else:
            self.metadata["train_config_path"]=f"{self.SUBMISSION_PATH}configs/train/T.yaml"
        self.metadata["mode"]="NAS"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader=train_loader
        self.valid_loader=valid_loader

        self.zcost_nas=False
        if "T0" in mode:
            self.zcost_nas=True #If using regressor for selecting best model instead of NAS
            self.ENAS=False # Use Evolutionary NAS from generation 2
            self.proxy=False # Use regressor to generate first population
            self.use_stages_pool=False # Transfer weights to models
            self.pretrained_pool=False # Use the pretrained_pool for the current pool
            self.initial_population_size=120
            self.update_pool=False
        elif "T1" in mode:
            self.ENAS=False # Use Evolutionary NAS from generation 2
            self.proxy=False # Use regressor to generate first population
            self.use_stages_pool=False # Transfer weights to models
            self.pretrained_pool=False # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=False # Add stem from first generation and add trained models to pool
        elif "T2" in mode:
            self.ENAS=True # Use Evolutionary NAS from generation 2
            self.proxy=False # Use regressor to generate first population     
            self.use_stages_pool=False # Transfer weights to models
            self.pretrained_pool=False # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=False # Add stem from first generation and add trained models to pool
        elif "T3" in mode:
            self.ENAS=True # Use Evolutionary NAS from generation 2
            self.proxy=True # Use regressor to generate first population
            self.use_stages_pool=False # Transfer weights to models
            self.pretrained_pool=False # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=False # Add stem from first generation and add trained models to pool
        elif "T4" in mode:
            self.ENAS=True # Use Evolutionary NAS from generation 2
            self.proxy=False # Use regressor to generate first population
            self.use_stages_pool=True # Transfer weights to models
            self.pretrained_pool=True # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=True # Add stem from first generation and add trained models to pool
        elif "T5" in mode:
            self.ENAS=True # Use Evolutionary NAS from generation 2
            self.proxy=True # Use regressor to generate first population
            self.use_stages_pool=True # Transfer weights to models
            self.pretrained_pool=True # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=False # Add stem from first generation and add trained models to pool
        elif "T6" in mode:
            self.ENAS=True # Use Evolutionary NAS from generation 2
            self.proxy=True # Use regressor to generate first population
            self.use_stages_pool=True # Transfer weights to models
            self.pretrained_pool=True # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=True # Add stem from first generation and add trained models to pool
        elif "T7" in mode:
            self.ENAS=True # Use Evolutionary NAS from generation 2
            self.proxy=True # Use regressor to generate first population
            self.use_stages_pool=True # Transfer weights to models
            self.pretrained_pool=True # Use the pretrained_pool for the current pool
            self.initial_population_size=20
            self.update_pool=True # Add stem from first generation and add trained models to pool
        ic(f"Mode {mode}")

        self.multiprocessing=False
        #if self.multiprocessing:
        current_method = mp.get_start_method(allow_none=True)
        print(current_method)
        if current_method!="spawn":
                nvmlInit()
                # Set the start method if it hasn't been set yet
                mp.set_start_method("spawn")
            
        #if self.metadata["codename"] in ["Caitie"]:
        #    ic(self.metadata["codename"])
        
        self.population_size=20
        self.total_generations=3 if (get_gpu_memory(0) / (1024 ** 3)) > 15.0 else 1
        ic(get_gpu_memory(0))
        ic(self.total_generations)
        self.num_best_parents=5
        self.sim_threshold=0.1

        ic("Time remaining:")
        ic(metadata['time_remaining'])


        self.study_name=f"tests_{metadata['codename']}_{current_date}"
        self.test_folder=f"{SAVE_PATH}/{self.study_name}"
        self.current_gen=1
        
        #Pretrained pool
        self.pool_stages_df=self.load_stages_pool(
            pool_folders= [pretrained_pool_path]) if self.pretrained_pool else pd.DataFrame()

        #self.pool_stages_df=self.load_stages_pool(
        #    pool_folders= ["/home/woody/iwb3/iwb3021h/NAS_COMPETITION_RESULTS/classifier_train/",
        #    "/home/woody/iwb3/iwb3021h/NAS_COMPETITION_RESULTS/stages_pool/"]) if self.pretrained_pool else pd.DataFrame()

        #####
        self.weights_pool={}
        self.best_models_info=pd.DataFrame()
        self.best_model={"score": 0,
                        "model_path": None,
                        "name": "",
                        "gen":0}
        self.old_chromosomes=[]
        self.parents=[]
        self.best_parents=pd.DataFrame()
        self.resume=False
        self.total_time=0
        self.current_time=time.time()
        if resume_from is not None:
            self.test_folder=resume_from
            
            file_path=f"{self.test_folder}/log.json"
            with open(file_path, 'r') as file:
                log = json.load(file)
            self.current_gen=log["generation"]
            self.current_model=log["current_model"]
            self.total_time=log["total_time"]
            self.resume=True
            if self.current_gen>1:
                self._load_backup()

    def search(self):   
        ic(self.zcost_nas)
        if self.zcost_nas:
            best_models,best_croms=self.regnet_space.create_first_generation(save_folder=None,gen=None, size=1, config_updates=None, metadata=self.metadata, samples=self.initial_population_size)
            ic(best_models.keys())
            ic(self.initial_population_size)

            self.metadata["train_config_path"]=f"{self.SUBMISSION_PATH}configs/train/finetuning_generation_adam.yaml"
            return best_models[list(best_models.keys())[0]]
        else:  
            while self.current_gen<self.total_generations+1:
                
                if self.resume:
                    models, chromosomes =self.regnet_space.load_generation(folder=f"{self.test_folder}/Generation_{self.current_gen}")

                else:
                    if self.current_gen==1:
                        if self.proxy:
                            ic("creating first generation")
                            models, chromosomes=self.regnet_space.create_first_generation(save_folder=self.test_folder,gen=self.current_gen, size=self.initial_population_size, config_updates=None, metadata=self.metadata)
                            ic(self.regnet_space.cfg)
                        else:
                            models, chromosomes= self.regnet_space.create_random_generation( 
                                                                                            save_folder=self.test_folder,
                                                                                            gen=self.current_gen,
                                                                                            size=self.population_size,
                                                                                            config_updates=None)
                    else:
                        ic(self.regnet_space.cfg)
                        offsprings_chromosomes=self.breeding(self.best_parents, self.population_size)
                        ic(self.regnet_space.cfg)
                        self._save_backup()
                        ic(self.regnet_space.cfg)
                        if self.ENAS:
                            ic(self.regnet_space.cfg)
                            models, chromosomes=self.regnet_space.create_generation(offsprings_chromosomes,
                                                                                    save_folder=self.test_folder,
                                                                                    gen=self.current_gen)
                        else:
                            models, chromosomes= self.regnet_space.create_random_generation( 
                                                                                            save_folder=self.test_folder,
                                                                                            gen=self.current_gen,
                                                                                            size=self.population_size,
                                                                                            config_updates=None)
                        
                    # Weights initialization
                    if self.use_stages_pool:
                        ic("start transfer weights")
                        models= self.transfer_weights(models, chromosomes )
                        if self.update_pool:
                            self.update_stages_pool(chromosomes)
                        ic(self.regnet_space.cfg)

                create_widths_plot(chromosomes).write_html(f"{self.test_folder}/Generation_{self.current_gen}/population.html")
                ic(self.regnet_space.cfg)
                generation_df, corr=self.train_generation(models, chromosomes)
                ic(self.regnet_space.cfg)
                self.best_parents=self.selection(generation_df)
                ic(self.regnet_space.cfg)
                self._save_backup()
                ic(self.regnet_space.cfg)
                self.current_gen+=1
                self.sim_threshold=self.sim_threshold-0.01
                
                ic(self.total_time)
                ic(self.metadata["time_remaining"])
                if self.total_time>14400 and self.current_gen==2:
                    self.total_generations=1
                if self.metadata["time_remaining"]<2*self.total_time and self.current_gen==2:
                    self.total_generations=1
            self.export_results()
            
            weights_file=self.best_model["model_path"]
            ind_path = weights_file.rfind('/')
            config_file = weights_file[:ind_path]
            best_model,info=self.regnet_space.load_model(config_file=f"{config_file}/config.yaml",
                                            weights_file=weights_file)
            self.metadata["train_config_path"]=f"{self.SUBMISSION_PATH}configs/train/finetuning_generation_adam.yaml"
            return best_model
    
    def train_mp(self,model,student):
        
        clear_output(wait=True)
        self.metadata["experiment_name"]=f"{self.test_folder}/Generation_{self.current_gen}/{student}"
        trainer=TrainerDistillation(model, self.device, self.train_loader, self.valid_loader, self.metadata, self.test) 
        trainer.train()
        del trainer, model, student
        torch.cuda.empty_cache()
        gc.collect()

        
        
    def train_generation(self,models,chromosomes):
        
        train_cfg=get_cfg()
        train_cfg.merge_from_file(self.metadata["train_config_path"])
        
        output_file_path = f"{self.test_folder}/Generation_{self.current_gen}/config.yaml"
        with open(output_file_path, "w") as f:
                f.write(train_cfg.dump()) 
        
        models_names=sorted(list(models.keys()))[:]     
        if self.resume:
            if self.current_model!="":
                 idx=models_names.index(self.current_model)                  
                 models_names=models_names[idx:]
                 if os.path.exists(f"{self.test_folder}/Generation_{self.current_gen}/{self.current_model}/worklog.txt"):
                    os.remove(f"{self.test_folder}/Generation_{self.current_gen}/{self.current_model}/worklog.txt")
                 self.resume=False
            else:
                models_names=[]
                
        
        if not self.multiprocessing:
            for student in models_names:
                clear_output(wait=True)
                print(student)

                self.total_time=time.time()-self.current_time+self.total_time
                self.current_time=time.time()
                with open(f"{self.test_folder}/log.json", 'w') as json_file:
                  json.dump({"state":"train","generation":self.current_gen,"current_model":student,"total_time":self.total_time},json_file )

                self.metadata["experiment_name"]=f"{self.test_folder}/Generation_{self.current_gen}/{student}"
                trainer=TrainerDistillation(models[student], self.device, self.train_loader, self.valid_loader, self.metadata, self.test) 
                trainer.train()
                torch.cuda.empty_cache()
                gc.collect()
        else:
                torch.cuda.empty_cache()
                next_process_index = 0
                ic("initial memory")
                print(f"Gpu free memory: {get_gpu_memory(0) / (1024 ** 3):.3f} GB")
                if self.metadata["codename"]=="Caitie":
                    required_memory = 33 * (2 ** 30)
                if self.metadata["codename"]=="Sadie":
                    required_memory = 15 * (2 ** 30)
                else:
                    required_memory = 7 * (2 ** 30)
                self.total_time=time.time()-self.current_time+self.total_time
                self.current_time=time.time()
                with open(f"{self.test_folder}/log.json", 'w') as json_file:
                  json.dump({"state":"train","generation":self.current_gen,"current_model":models_names[0],"total_time":self.total_time},json_file )

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
                        p = mp.Process(target=self.train_mp, args=(models[student],student))
                        p.start()
                        processes.append(p)
                        next_process_index += 1
                        torch.cuda.empty_cache()
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
                    torch.cuda.empty_cache()
            

        self.total_time=time.time()-self.current_time+self.total_time
        self.current_time=time.time()
        with open(f"{self.test_folder}/log.json", 'w') as json_file:
            json.dump({"current_model":"", "generation":self.current_gen, "total_time":self.total_time},json_file )
        
        if models_names is not None:
            return get_generation_dfs(f"{self.test_folder}/Generation_{self.current_gen}", corr=True, chromosomes=chromosomes, save=True, gen=self.current_gen)
        else:
            return get_generation_dfs(f"{self.test_folder}/Generation_{self.current_gen}", corr=False, chromosomes=chromosomes, save=False, gen=self.current_gen)

    def selection(self,df):
        df=df.sort_values("best_acc", ascending=False)
        self.old_chromosomes=self.old_chromosomes+df[["WA","W0","WM","DEPTH"]].values.tolist()
        if len(self.best_models_info!=0):
            self.best_models_info=pd.concat([self.best_models_info,df.head(1)])
        else:
            self.best_models_info=df.head(1)
        
        best_new_score=df.head(1).iloc[0]["best_acc"]
        print("Best new score:", best_new_score)
        if best_new_score>=self.best_model["score"]:
            new_name=df.head(1).iloc[0]["name"]
            self.best_model={
                "score":best_new_score,
                "model_path": f"{self.test_folder}/Generation_{self.current_gen}/{new_name}/student_best",
                "name": new_name,
                "gen":self.current_gen
            }
        return df.head(self.num_best_parents)[["name","best_acc","WA","W0","WM","DEPTH", "GROUP_W"]].reset_index(drop=True)

    def breeding(self,best_parents, n_children=20):
        parents_names=best_parents["name"].values.tolist()
        parent_names_combinations = list(itertools.combinations(parents_names, 2))
        
        
        parents=best_parents[["WA","W0","WM","DEPTH"]].values.tolist()
        parent_combinations = list(itertools.combinations(parents, 2))
        children=[]
        for parent_1, parent_2 in parent_combinations:
            children.append(self.crossover(parent_1, parent_2, mode="mean"))


        for parent_1, parent_2 in parent_combinations:
            children.append(self.crossover(parent_1, parent_2, mode="one_point"))
        
        parent_names_combinations=parent_names_combinations+parent_names_combinations
        parent_combinations=parent_combinations+parent_combinations
        
        child_names=[]
        for i in range(len(children)):
            child_names.append(generate_slug(2).replace("-", "_"))
        child_names=sorted(child_names)
        children_dict={}
        
        for idx, child in enumerate(children):
            child_diff=0
            while child_diff<self.sim_threshold:
                for old_chrom in self.old_chromosomes+children[:idx]:
                    child_diff=self.regnet_space.compare_chromosomes(child,old_chrom)
                    if child_diff<0.1:
                        print("Child:", child)
                        print("Old_chrom:", old_chrom)
                        print("Diff:", child_diff)
                        child=self.mutation(child)
                        print(child)
                        children[idx]=child
                        break
            children_dict[child_names[idx]]=child
            self.parents.append({"parents_names":str(parent_names_combinations[idx]),
                                "parents":str(parent_combinations[idx]),
                                "child_name":str(child_names[idx]),
                               "child":str(child),
                               "generation": int(self.current_gen)})
        return children_dict

    def mutation(self,chrom, dwa=8, dwm=0.05, dd=1, dw0=8):
        max_range=[max(self.regnet_space.WA_OPTIONS),max(self.regnet_space.W0_OPTIONS),max(self.regnet_space.WM_OPTIONS), max(self.regnet_space.D_OPTIONS)]
        min_range=[min(self.regnet_space.WA_OPTIONS),min(self.regnet_space.W0_OPTIONS),min(self.regnet_space.WM_OPTIONS),min(self.regnet_space.D_OPTIONS)]
        wa,w0,wm,d=chrom
        
        dwa=random.choice([-1,0,1])*dwa
        wa=wa+dwa
        wa=max(wa,min_range[0])
        wa=min(wa,max_range[0]+16)

        dw0=random.choice([-1,0,1])*dw0
        w0=w0+dw0
        w0=max(w0,min_range[1])
        
        #while wm>=min_range[2] and wm<=max_range[2]:
        dwm=random.choice([-1,0,1])*dwm
        wm=wm+dwm
        wm=max(wm,min_range[2])
        wm=min(wm,max_range[2]+0.1)
        
        #while d>=min_range[3] and d<=max_range[3]:
        dd=random.choice([-1,0,1])*dd
        d=d+dd
        d=max(d,min_range[3])
        d=min(d,max_range[3])
       
        if w0<wa:
            w0=int(random.choice([option for option in list(self.regnet_space.W0_OPTIONS) if option >= wa]))

        return [wa,w0,wm,d]

    def crossover(self,p1, p2, mode):
        def mean_round_n(a,b,n):
            r=(a+b)/2
            return np.ceil(r/ n) * n
        if mode=="mean":
            wa=float(mean_round_n(p1[0],p2[0],2))
            w0=int(mean_round_n(p1[1],p2[1],self.regnet_space.W0_STEP))
            if w0<wa:
                w0=int(random.choice([option for option in self.regnet_space.W0_OPTIONS if option >= wa]))
            wm=float(mean_round_n(p1[2],p2[2],self.regnet_space.WM_STEP))
            d=int(mean_round_n(p1[3],p2[3],self.regnet_space.D_STEP)) 

        elif mode=="one_point":
            order=random.choice([0,1,2,3])
            if order==0:
                wa,w0,wm,d=p1[0], p2[1], p2[2], p1[3]
            elif order==1:
                wa,w0,wm,d=p2[0], p1[1], p1[2], p2[3]
            elif order==2:
                wa,w0,wm,d=p1[0], p1[1], p2[2], p2[3]
            elif order==3:
                wa,w0,wm,d=p2[0], p2[1], p1[2], p1[3]
        return [wa,w0,wm,d]
    
    def transfer_weights(self, models, chromosomes):
        #WHOLE MODELS INHERITANCE LOOP
        if self.pool_stages_df.empty:
            return models

        ic("load chromosomes")
        #WHOLE LOOP SELECTION PRETRAINED INDIVIDUALS
        df_models=pd.DataFrame(chromosomes).T[["ws","ds","num_stages", "DEPTH"]]
        total_pool_individuals={}

        for model_name in list(chromosomes.keys()):
            df_current_model=df_models.loc[model_name]
            ic(model_name)
            filtered_dfs=[]
            df_results_aux=self.pool_stages_df.drop(columns=["ws","ds"])
            df_results_aux["diff_stages"]=abs(df_results_aux["num_stages"]-df_current_model["num_stages"])
            df_results_aux["diff_depth"]=abs(df_results_aux["DEPTH"]-df_current_model["DEPTH"])

            for stage in range(1, df_current_model["num_stages"]+1):
                df_results_aux[f"diff_ws{stage}"]=abs(df_results_aux[f"ws{stage}"]-df_current_model["ws"][stage-1])
                df_results_aux[f"diff_d{stage}"]=abs(df_results_aux[f"ds{stage}"]-df_current_model["ds"][stage-1])

            for stage in range(1, df_current_model["num_stages"]+1):
                if stage==1:
                    df_results_aux=df_results_aux.sort_values(["diff_ws1","diff_d1","diff_stages","diff_ws2","diff_depth"])
                else:
                    df_results_aux=df_results_aux.sort_values([f"diff_ws{stage}",f"diff_d{stage}",f"diff_ws{stage-1}","diff_stages", "diff_depth"])

                if stage==1:
                    first_row_values = df_results_aux[["diff_stages", f"diff_ws{stage}", f"diff_d{stage}"]].iloc[0]
                    # Filter the DataFrame based on these values
                    filtered_df = df_results_aux[
                        (df_results_aux["diff_stages"] == first_row_values["diff_stages"]) &
                        (df_results_aux[f"diff_ws{stage}"] == first_row_values[f"diff_ws{stage}"]) &
                        (df_results_aux[f"diff_d{stage}"] == first_row_values[f"diff_d{stage}"])
                    ]
                else:
                    first_row_values = df_results_aux[["diff_stages",f"diff_ws{stage-1}", f"diff_ws{stage}", f"diff_d{stage}"]].iloc[0]
                    # Filter the DataFrame based on these values
                    filtered_df = df_results_aux[
                        (df_results_aux["diff_stages"] == first_row_values["diff_stages"]) &
                        (df_results_aux[f"diff_ws{stage-1}"] == first_row_values[f"diff_ws{stage-1}"]) &
                        (df_results_aux[f"diff_ws{stage}"] == first_row_values[f"diff_ws{stage}"]) &
                        (df_results_aux[f"diff_d{stage}"] == first_row_values[f"diff_d{stage}"])
                    ]
                filtered_dfs.append(filtered_df)

            pool_individuals={}
            items=[]
            for idx, stage_df in enumerate(filtered_dfs):
                items.append(dict(zip(stage_df.index.tolist(),stage_df.dataset.tolist())))
            for idx, item in enumerate(items):
                for i in range(0,len(items)):
                    if i !=idx:
                        common_items = item.items() & items[i].items()
                        #print(common_items)
                        if common_items:
                            pool_individuals[idx+1]=next(iter(common_items))
                            break
                if idx+1 not in pool_individuals:
                     pool_individuals[idx+1]=next(iter(item.items()))
            ic("########################")

            total_pool_individuals[model_name]=pool_individuals    

        with open('individuals_pool.json', 'w') as f:
            json.dump(total_pool_individuals, f)

        n_access={}
        for model_name in list(models.keys()):
            ic("Model Name: ",model_name)
            ic("#######################")
            pool_models={}
            pool_chroms={}
            for stage, info in total_pool_individuals[model_name].items():
                name, transfer_dataset=info
                ic(name)
                ic(transfer_dataset)
                weights_file=f"{transfer_dataset}/{name}/student_best"
                config_file=f"{transfer_dataset}/{name}/config.yaml"
                if os.path.exists(weights_file):
                    pool_models[stage],pool_chroms[stage]=self.regnet_space.load_model(config_file=config_file, weights_file=weights_file)
                else:
                    ic(f"Weights file does not exits {weights_file}")
                    pool_chroms[stage]=None
            chrom=chromosomes[model_name]
            n_access[model_name]=0
            for stage in range(1,chrom["num_stages"]+1):
                if pool_chroms[stage] is not None:
                    max_block=min(chrom["ds"][stage-1], pool_chroms[stage]["ds"][stage-1])
                    ic("###### MAX BLOCK #####: ",max_block)
                    for block in range(1,max_block+1):
                        ic("Block: ", block)
                        model_part = eval(f"models[model_name].s{stage}.b{block}")
                        orig_part = eval(f"pool_models[stage].s{stage}.b{block}.state_dict()")

                        for key in model_part.state_dict().keys():

                            tensor = orig_part[key]
                            tensor_shape = tensor.shape

                            tensor_student_shape=model_part.state_dict()[key].shape
                            if tensor_shape==tensor_student_shape:
                                print(key)
                                #print(tensor_shape)
                                n_access[model_name]=n_access[model_name]+1


                                keys = key.split('.')

                                # Access the specific layer that contains the weight attribute
                                param = functools.reduce(getattr, keys[:-1], model_part)
                                #print(param.requires_grad)
                                #param.weight.requires_grad=False
                                # Use setattr to update the .data attribute of the weight tensor
                                getattr(param, keys[-1]).data = tensor.clone()
                else:
                    ic(f"Stage {stage} is None")
        pd.DataFrame([n_access]).T.sort_values(by=0).to_csv("n_access.csv")

        if self.current_gen>1 and self.update_pool:
            weights_file=self.best_model["model_path"]
            ind_path = weights_file.rfind('/')
            config_file = weights_file[:ind_path]
            best_model,info=self.regnet_space.load_model(config_file=f"{config_file}/config.yaml",
                                            weights_file=weights_file)

            for model_name in list(models.keys()):
                models[model_name].stem.load_state_dict(best_model.stem.state_dict())

        self.regnet_space=RegNet(self.metadata,
                    W0=[16, 120, 8],
                    WA=[16, 64, 8],
                    WM=[2.05,2.9,0.05],
                    D=[8,22,1], 
                    G=[8,8,8], 
                    base_config=f"{self.SUBMISSION_PATH}configs/search_space/config.yaml")
        return models
    
    def load_stages_pool(self, pool_folders):
        df_results_list=[]
        for pool_folder in pool_folders:
            df_results=pd.read_csv(f"{pool_folder}/df_blocks_pool.csv", index_col=0)
            df_results=df_results[df_results.dataset!=self.metadata["codename"]]
            df_results["dataset"]=pool_folder+df_results["dataset"]
            df_results_list.append(df_results)
        df_results=pd.concat(df_results_list)
        df_results.to_csv("initial_pool.csv")
        return df_results
        
    def update_stages_pool(self,chromosomes):
        ic("update_stages_pool")
        df_results=pd.DataFrame(chromosomes).T[["ws","ds","num_stages", "DEPTH"]].reset_index()
        df_results["dataset"]=f"{self.test_folder}/Generation_{self.current_gen}"
        df_results.to_csv(f"{self.current_gen}_gen_df.csv")
        for idx, row in df_results.iterrows():
            #for i, w in enumerate(ast.literal_eval(row["ws"])):
            for i, w in enumerate(row["ws"]):
                ic(idx, f"ws{i+1}")
                df_results.at[idx, f"ws{i+1}"] = int(w)

        for col in ["ws1","ws2","ws3","ws4","ws5"]:
            if col not in df_results.columns:
                df_results[col]=0
            else:
                df_results[col]=df_results[col].fillna(0).astype(int)

        #########################################################

        for idx, row in df_results.iterrows():
            for i, w in enumerate(row["ds"]):
            #for i, w in enumerate(ast.literal_eval(row["ds"])):
                df_results.at[idx, f"ds{i+1}"] = int(w)

        for col in ["ds1","ds2","ds3","ds4","ds5"]:
            if col not in df_results.columns:
                df_results[col]=0
            else:
                df_results[col]=df_results[col].fillna(0).astype(int)
        ic(self.pool_stages_df.empty)
        self.pool_stages_df = pd.concat([self.pool_stages_df, df_results.set_index("index")]) if not self.pool_stages_df.empty else df_results
        self.pool_stages_df.to_csv(f"updated_pool_generation_{self.current_gen}.csv")
        
    
    def _save_backup(self):
        log={}
        log["old_chromosomes"]=self.old_chromosomes
        log["best_model"]=self.best_model
        log["best_models_info"]=self.best_models_info.to_json()
        log["best_parents"]=self.best_parents.to_json()
        log["sim_threshold"]=self.sim_threshold
    
        parents_df=pd.DataFrame(self.parents)
        log["parents"]=self.parents
        
        if not os.path.exists(f"{self.test_folder}/Generation_{self.current_gen}"):
            os.makedirs(f"{self.test_folder}/Generation_{self.current_gen}")
        if self.current_gen>1:
            parents_df[parents_df.generation==self.current_gen].to_csv(f"{self.test_folder}/Generation_{self.current_gen}/parents.csv")
        
        
        with open(f"{self.test_folder}/log.evonas", 'w') as json_file:
          json.dump(log, json_file)
    def _load_backup(self):
        file_path=f"{self.test_folder}/log.evonas"
        with open(file_path, 'r') as file:
            log = json.load(file)
        self.best_models_info=pd.read_json(StringIO(log["best_models_info"]))
        self.best_model=log["best_model"]
        self.old_chromosomes=log["old_chromosomes"]
        self.best_parents=log["best_parents"]
        self.parents=log["parents"]
        self.sim_threshold=log["sim_threshold"]

    def export_results(self):
        results_file={}
        results_file["correlation"]=[]
        results_file["results"]=[]
        results_file["parents"]=[]

        for gen in range(1,self.current_gen):
            print(gen)
            #with open(f"{self.test_folder}/Generation_{gen}/corr.txt", 'r') as file:
            #    content = file.read()
            #    corr = ast.literal_eval(content)
            #    results_file["correlation"].append(pd.DataFrame(corr, columns=[gen]).T.reset_index(names="generation"))
            df_r=pd.read_csv(f"{self.test_folder}/Generation_{gen}/results.csv", index_col=0)
            df_r["generation"]=gen
            results_file["results"].append(df_r)
            if gen!=1:
                results_file["parents"].append(pd.read_csv(f"{self.test_folder}/Generation_{gen}/parents.csv", index_col=0))

        #results_file["correlation"]=pd.concat(results_file["correlation"]).reset_index(drop=True).to_json()
        results_file["results"]=pd.concat(results_file["results"]).reset_index(drop=True).to_json()
        if len(results_file["parents"])>0:
            results_file["parents"]=pd.concat(results_file["parents"]).reset_index(drop=True).to_json()
        results_file["metadata"]=self.metadata
        results_file["best_model"]=self.best_model
        results_file["parameters"]={"ENAS":self.ENAS,
                                    "proxy":self.proxy,
                                    "transfer_weights": self.use_stages_pool,
                                    "pretrained_pool": self.pretrained_pool,
                                   "population_size":self.population_size,
                                   "total_generations":self.total_generations,
                                   "num_best_parents": self.num_best_parents,
                                   "sim_threshold":self.sim_threshold,
                                    "multiprocessing":self.multiprocessing,
                                    "update_pool":self.update_pool,
                                    "initial_population_size":self.initial_population_size
                                  }
        results_file["total_time"]=self.total_time

        with open(f"{self.test_folder}/{self.study_name}.evonas", 'w') as json_file:
                json.dump(results_file, json_file)
        
        
