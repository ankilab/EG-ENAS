import pandas as pd
import plotly.graph_objects as go
import numpy as np
import torch
import os
def compute_model_size(model):
    """
    Computes the size of the PyTorch model in megabytes (MB) based on the sizes of its parameters.

    Args:
        model (torch.nn.Module): The PyTorch model for which to compute the size.

    Returns:
        float: The size of the model in megabytes (MB).
    """
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # Convert size to megabytes (optional)
    total_size_mb = total_size / (1024 ** 2)
    return  total_size_mb

def load_checkpoint(path):
    """
    Loads a PyTorch model checkpoint from the specified file path.

    Args:
        path (str): The file path to the checkpoint file.

    Returns:
        dict or torch.nn.Module: The loaded checkpoint. If it's a dictionary, it typically contains
        the model state_dict along with other information like optimizer state, epoch, etc.
        If it's a torch.nn.Module, it's just the model itself.
    """
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
    

def create_widths_plot(chromosomes):
    """
    Creates a plot showing the widths of blocks for each chromosome in the population.

    Args:
        chromosomes (dict): A dictionary where keys represent chromosome names and values represent
            information about the chromosome, including block widths ('ws') and block depths ('ds'),.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure displaying the widths of blocks for each chromosome.
    """
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
        title='Widths per block',
        xaxis=dict(title='Block index'),
        yaxis=dict(title="Width"),
        template="presentation"
    )
    fig.layout=layout
    fig.update_layout(height=700, width=900)
    return fig


def scatter_results(chromosomes,columny,columnx="FLOPS",results_path=None):
    """
    Creates a scatter plot of FLOPS (Floating Point Operations Per Second) versus a specified column's values
    for each chromosome.

    Args:
        chromosomes (dict): A dictionary where keys represent chromosome names and values represent
            information about the chromosome, including FLOPS ('flops'), WA ('Width slope'), W0 ('Initial width'), WM ('Width multiplier'), DEPTH ('Total depth')m.
        column (str): The name of the column from the chromosome information to plot against FLOPS.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure displaying a scatter plot of FLOPS versus
            the specified column's values for each chromosome.

    """
    chromosomes_df=pd.DataFrame(chromosomes).T.reset_index().rename(columns={"index":"name"})
    if results_path is not None:
        results_df=pd.read_csv(results_path, index_col=0)[["name","best_acc","epoch_time"]]
        chromosomes_df=pd.merge(chromosomes_df,results_df, on="name", how="left")
    fig = go.Figure(data=go.Scatter(
        x=chromosomes_df[columnx],
        y=chromosomes_df[columny],
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
        xaxis=dict(title=columnx),
        yaxis=dict(title=columny),
        template="presentation"
    )
    fig.layout=layout
    return fig


def results_to_df(path, name):
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
            elif 'best_acc' in line or "Total time" in line:
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
    df=df.assign(name=name)
    
    return df

def get_generation_dfs(folder, corr=False, chromosomes=None, save=False):
    students=os.listdir(folder)
    print(students)
    sudents=[student for student in students if "ipynb_checkpoints" not in student]
    students_df=[]
    for student in students:
        try:
            students_df.append(results_to_df(f"{folder}/{student}/worklog.txt", student))
        except:
            pass
    epochs=len(students_df[0])
    students_df=pd.concat(students_df, ignore_index=True)
    students_df=students_df.sort_values(by=["epoch","test_acc"], ascending=[False,False])
    students_df=students_df.assign(study="vainilla")
    
    idx = students_df.groupby("name")["test_acc"].idxmax()
    #idx = students_df.groupby("name")["epoch"].idxmax()
    max_test_acc_rows = students_df.loc[idx]
    sorted_df=max_test_acc_rows.sort_values(by="test_acc")
    sorted_df=sorted_df.rename(columns={"test_acc":"best_acc"})
    order_students=list(sorted_df.name.values)

    vainilla_students=[]
    students_df["name"] = pd.Categorical(students_df["name"], categories=order_students, ordered=True)
    for i in range(1,epochs+1):
        vainilla_students.append(students_df[students_df.epoch==i].sort_values(by="name")) 
    
    corr_coeff_vainilla=[]
    if corr:
        for epoch in vainilla_students:
            correlation_coefficient =np.corrcoef(list(epoch['test_acc'].values),list(sorted_df['best_acc'].values))
            corr_coeff_vainilla.append(correlation_coefficient[0,1])
            
    if chromosomes is not None:
        chromosomes_df=pd.DataFrame(chromosomes).T.reset_index().rename(columns={"index":"name"})
        sorted_df=pd.merge(chromosomes_df[['name', 'ws', 'ds', 'num_stages', 'params','WA', 'W0', 'WM', 'DEPTH', 'GROUP_W']],sorted_df, on="name", how="left")
        
    if save:
        sorted_df.to_csv(f"{folder}/results.csv")
        with open(f"{folder}/corr.txt", 'a') as file:
           file.write(str(corr_coeff_vainilla))
    return sorted_df, corr_coeff_vainilla

        