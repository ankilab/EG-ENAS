import pandas as pd

def compute_model_size(model):
    """
    Computes the size of the PyTorch model in megabytes (MB) based on the sizes of its parameters.

    Args:
        model (torch.nn.Module): The PyTorch model for which to compute the size.

    Returns:
        float: The size of the model in megabytes (MB).
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # Convert size to megabytes (optional)
    total_size_mb = total_size / (1024 ** 2)
    return  total_size_mb
    """

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


def scatter_flops(chromosomes,column):
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