import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np 
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import colour


def paperStyle(font_size=8, use_seaborn=True, temporary=True, ticks='out'):
    """Defines plot styles for paper

    Args:
        font_size (int, optional): Figure font size. Defaults to 8 (pt).
        use_seaborn (bool, optional): If seaborn is used to style the plot. Defaults to True.
        temporary (bool, optional): Use `paperStyle` with `with` statement. Defaults to True.
    """    
    if use_seaborn:
        sns.set_style('white')
        sns.set_style('ticks')

    if temporary:
        return mpl.rc_context({
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
            'axes.titlesize': font_size,
            'xtick.direction': ticks,
            'ytick.direction': ticks,
            'font.family': ['sans-serif'],
            'font.sans-serif': ['Arial'],
            'svg.fonttype': 'none',
            'pdf.fonttype': 42

        })

    else:
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['xtick.labelsize'] = font_size
        plt.rcParams['ytick.labelsize'] = font_size
        plt.rcParams['legend.fontsize'] = font_size
        plt.rcParams['axes.titlesize'] = font_size
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['xtick.direction'] = ticks
        plt.rcParams['ytick.direction'] = ticks
        plt.rcParams['svg.fonttype'] = 'none' # Text is not rendered
        plt.rcParams['pdf.fonttype'] = 42 # TrueType to avoid PDF issues
    

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
    df=df.assign(name=name)
    
    return df

def get_predictor_data(studies):
    train_data=[]
    for study_folder in studies: 
        data=[]
        #study_folder=f"/home/woody/iwb3/iwb3021h/NAS_COMPETITION_RESULTS/classifier_train/{study_name}"
        subjects= os.listdir(study_folder)

        subjects=[subject for subject in subjects if os.path.isdir(f"{study_folder}/{subject}")]
        print(subjects)
        for name in subjects:
            try:
                test_acc=results_to_df(f"{study_folder}/{name}/worklog.txt", f"{name}")[["epoch","test_acc","name"]]
                #test_acc=test_acc.rename(columns={"test_acc":"acc","test_loss":"test_acc"})
                #test_acc=test_acc.rename(columns={"test_loss":"test_acc"})
                #print(test_acc.columns)
                #if len(test_acc)>49:
                test_acc_piv = test_acc[["epoch","test_acc","name"]].pivot(index='name', columns='epoch', values='test_acc').add_prefix("epoch_").reset_index()
                #test_acc_piv["best_acc"]=test_acc["acc"].max()
                data.append(test_acc_piv)
            except:
                pass
        data=pd.concat(data)
        train_data.append(data)
    train_data=pd.concat(train_data)
    return train_data


def get_train_data(main_path, test_name, test_path, num_generations):
    train_data_list = []

    for gen in range(1, num_generations + 1):
        train_data_gen = get_predictor_data(studies=[f"{main_path}/{test_name}/{test_path}/Generation_{gen}"])
        train_data_gen["generation"] = gen
        train_data_list.append(train_data_gen)

    train_data = pd.concat(train_data_list).set_index(["name", "generation"])

    train_data_cum = train_data.cummax(axis=1)
    train_data_cum["best_acc"] = train_data_cum.max(axis=1)
    train_data["best_acc"] = train_data.max(axis=1)

    return train_data, train_data_cum


def plot_boxplots(df_list, x_column, y_column, colors, title='Box Plot per DataFrame', names=[]):
    fig = go.Figure()

    # Combine all y_column values from all DataFrames to calculate global percentiles
    combined_y_values = np.concatenate([df[y_column].values for df in df_list])

    # Calculate 25th percentile and max value
    y_min = np.percentile(combined_y_values, 2)
    y_max = np.max(combined_y_values)

    # Loop over each DataFrame and plot its box plot with the specified color
    for i, df in enumerate(df_list):
        fig.add_trace(go.Box(
            x=df[x_column],
            y=df[y_column],
            name=f"{names[i]}",
            marker_color=colors[i % len(colors)],  # Use specified colors
            #boxmean='sd'  # Show standard deviation mean lines
        ))

    # Customize layout for better presentation
    fig.update_layout(
        title=title,
        xaxis_title=x_column.capitalize(),
        yaxis_title=y_column.capitalize(),
        title_font_size=18,
        yaxis_title_font_size=16,
        xaxis_title_font_size=16,
        width=800,
        height=600,
        font=dict(size=14),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        template="presentation",
        boxmode='group'  # Group boxes for better comparison
    )

    # Set y-axis range between 25th percentile and max value
    fig.update_yaxes(range=[y_min, y_max+1])

    # Show the plot
    fig.show()
    return fig