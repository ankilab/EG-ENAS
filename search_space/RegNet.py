import os
import pycls.core.builders as builders
import pycls.core.checkpoint as cp
from pycls.core.config import cfg, load_cfg, reset_cfg
from coolname import generate_slug
import random
import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from utils import compute_model_size

class RegNet:
    def __init__(self, metadata, W0=[8, 56, 8], WA=[8, 48, 8],WM=[2.0,2.9,0.05],D=[6,20,1]):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        WA_OPTIONS=np.arange(WA[0],WA[1]+WA[2], WA[2])
        W0_OPTIONS=np.arange(W0[0],W0[1]+W0[2], W0[2])
        WM_OPTIONS=np.arange(WM[0],WM[1]+WM[2], WM[2])
        D_OPTIONS=np.arange(D[0],D[1]+D[2],D[2])
        self.cfg=cfg
############################################

    def adjust_block_compatibility(self, ws, bs, gs):
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

    def generate_regnet(self,w_a, w_0, w_m, d, q=8):
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

        def get_blocks_per_stage(self):
            ws, ds, num_stages, total_stages, ws_all, ws_cont=generate_regnet(w_a=self.REGNET.WA, w_0=self.REGNET.W0,    w_m=self.cfg.REGNET.WM, d=self.cfg.REGNET.DEPTH)
            ss = [self.cfg.REGNET.STRIDE for _ in ws]
            bs = [self.cfg.REGNET.BOT_MUL for _ in ws]
            gs = [self.cfg.REGNET.GROUP_W for _ in ws]
            ws, bs, gs = self.adjust_block_compatibility(ws, bs, gs)
            info={"ws":ws,"bs":bs,"gs":gs,"ds":ds,"num_stages":num_stages}
            return info


    def create_random_model(self, test_folder, config_list=None, save_config=False, random_name=None, gen=None):
        """Constructs a predefined model"""
        # Load the config
        if config_list is None:
            self.cfg.REGNET.WA=float(random.choice(self.WA_OPTIONS))
            self.cfg.REGNET.W0=int(random.choice(self.W0_OPTIONS))
            self.cfg.REGNET.WM=float(random.choice(self.WM_OPTIONS))
            self.cfg.REGNET.DEPTH=int(random.choice(self.D_OPTIONS)) 
        else:
            cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH = config_list
            #cfg.REGNET.GROUP_W=8

        # Write the dictionary to a YAML file
        if save_config:
            if random_name is None:
                random_name = generate_slug(2).replace("-", "_")
            print("Created model: ", random_name)


            if gen is not None:
                output_directory=f"{test_folder}/Generation_{gen}/{random_name}"
            else:
                output_directory=f"{test_folder}/{random_name}"

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            output_file_path = f"{output_directory}/config.yaml"
            with open(output_file_path, "w") as f:
              f.write(cfg.dump()) 

        # Construct model
        model=RegNet().to(self.device)
        model = builders.build_model().to(device)
        # Load pretrained weights
        info=get_blocks_per_stage()
        total_size_mb=compute_size(model)
        info["total_size_mb"]=total_size_mb
        info.update(model.complexity({"h":0, "w":0, "flops":0, "params":0, "acts":0}))
        info["WA"]=cfg.REGNET.WA
        info["W0"]=cfg.REGNET.W0
        info["WM"]=cfg.REGNET.WM
        info["DEPTH"]=cfg.REGNET.DEPTH
        return model, info


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
def load_model(config_file, weights_file=None):
    """Constructs a predefined model (note: loads global config as well)."""
    # Load the config
    reset_cfg()
    load_cfg(config_file)
    
    # Construct model
    model = builders.build_model().to(device)
    print("Loading model:", config_file)
    # Load pretrained weights
    if weights_file is not None:
        state = load_checkpoint(weights_file)
        model.load_state_dict(state["model"])
        
    info=get_blocks_per_stage()
    total_size_mb=compute_size(model)
    #info["total_params"]=total_params
    info["total_size_mb"]=total_size_mb
    info.update(model.complexity({"h":0, "w":0, "flops":0, "params":0, "acts":0}))
    info["WA"]=cfg.REGNET.WA
    info["W0"]=cfg.REGNET.W0
    info["WM"]=cfg.REGNET.WM
    info["DEPTH"]=cfg.REGNET.DEPTH
    return model, info, cfg

def create_random_generation(test_folder,gen, size):
    models={}
    chromosomes={}
    for ind in range(size):
        random_name = generate_slug(2).replace("-", "_")
        model, info=create_random_model(f"{test_folder}", save_config=True, random_name=random_name, gen=Gen)
        models[random_name]=model
        chromosomes[random_name]=info
    return models, chromosomes

def load_generation(folder):
    models={}
    chromosomes={}
    configs={}
    individuals=os.listdir(folder)
    individuals=[ind for ind in individuals if os.path.isdir(os.path.join(folder, ind))]
    for ind in individuals:
        ind_folder=f"{folder}/{ind}"
        models[ind], chromosomes[ind], configs[ind]=load_model(ind_folder)
    return models,chromosomes, configs


def create_widths_plot(chromosomes):
    fig = go.Figure()

    for ind, info in chromosomes.items():
        widths=np.repeat(info["ws"],info["ds"])
        block_index=np.arange(1,len(widths)+1)
        fig.add_trace(go.Scatter(x=block_index, y=widths,
                            mode='lines+markers',
                            marker=dict(
                                opacity=0.6,    # Opacity level (0.0 to 1.0)
                                line=dict(
                                    width=1,
                                   # opacity=0.5# Line width of markers
                                )),

                            name=ind+"_"+str(info["DEPTH"])+""))
    layout = go.Layout(
        title='Generation 1 Cifar100',
        xaxis=dict(title='Block index'),
        yaxis=dict(title="Width"),
        template="presentation"
    )
    fig.layout=layout
    fig.update_layout(height=700, width=900)
    return fig

def scatter_flops(chromosomes_df,column):
    fig = go.Figure(data=go.Scatter(
        x=chromosomes_df['flops']/1000000,
        y=chromosomes_df[column],
        mode='markers',
        #marker=dict(
        #    size=2,
            #color=np.random.randn(500), #set color equal to a variable
            #colorscale='Viridis', # one of plotly colorscales
            #showscale=True
        #),
        marker=dict(
            size=10
        ),
        text=chromosomes_df['name']
    ))
    layout = go.Layout(
        title='',
        xaxis=dict(title='FLOPS (Millions)'),
        yaxis=dict(title=column),
        template="presentation"
    )
    fig.layout=layout
    return fig