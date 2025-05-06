from .train_evaluate import get_evaluation_results_filename, get_diversity_results_filename
from .file_utils import load_picklefile, validate_folderpath


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
import plotly.offline as py

pd.options.plotting.backend = "plotly"
py.init_notebook_mode() # graphs charts inline (IPython).

a4_dims = (11.7, 8.27)


def plot_and_save_heatmap(df,
                          round_point=2,
                          title='Recall@20 for checkpoint models across Holdouts - model - data',
                          filepath='images/heatmaps/',
                          filename='<base_dataset_name><FILENAME_VERSION>_<model_name>_<VALID_METRIC>.<file extension>'):
    
    # code from Measuring Forgetting  
    plt.figure(figsize=(15, 10))
    x_t = np.arange(0, df.shape[0])
    labels=[str(i+1) for i in x_t]
    sns.heatmap(df, vmin=0, vmax=df.max().max(), annot=True, fmt=f'0.{round_point}f', linewidths=.1, cmap='Spectral_r', xticklabels=labels, yticklabels=labels)
    plt.ylabel('model')
    plt.xlabel('holdout')
    plt.title(title)
    if filepath:
        validate_folderpath(filepath)
        plt.savefig(filepath+filename);
    else:
        plt.show()



def get_full_models_results_matrix(model_name,
                       models_versions, 
                       base_dataset_name, 
                       save_path, 
                       metric, 
                       filename_version):
    
    cols = [i for i in range(len(models_versions))]
    df = pd.DataFrame(columns=cols) 

    
    for row_idx, model_ver in enumerate(models_versions):
        # print('\n\n', row_idx, model_ver)
        # print('\n\n',model_ver)
        current_dataset_name = base_dataset_name+model_ver

        for column_idx, section in enumerate(models_versions):
            file_dir = save_path+current_dataset_name+\
                                '/'+get_evaluation_results_filename(model_name,
                                                                    current_dataset_name,
                                                                    section,
                                                                    filename_version)+'.pkl'
            # print(file_dir, 'row_idx='+str(row_idx)+'; column_idx='+str(column_idx))
            test_recall = load_picklefile(file_dir)[metric]

            df.loc[row_idx,column_idx] = test_recall

    df = df.apply(pd.to_numeric, errors='coerce')  
    return df


def get_full_models_diversity_results_matrix(model_name,
                       models_versions, 
                       base_dataset_name, 
                       save_path, 
                       metric, 
                       filename_version):
    
    cols = [i for i in range(len(models_versions))]
    df = pd.DataFrame(columns=cols) 

    
    for row_idx, model_ver in enumerate(models_versions):
        # print('\n\n', row_idx, model_ver)
        # print('\n\n',model_ver)
        current_dataset_name = base_dataset_name+model_ver

        for column_idx, section in enumerate(models_versions):
            file_dir = save_path+current_dataset_name+\
                                '/'+get_diversity_results_filename(model_name,
                                                                    current_dataset_name,
                                                                    section,
                                                                    filename_version)+'.pkl'
            # print(file_dir, 'row_idx='+str(row_idx)+'; column_idx='+str(column_idx))
            test_diversity = load_picklefile(file_dir)[metric]
            # print(load_picklefile(file_dir))

            df.loc[row_idx,column_idx] = test_diversity

    df = df.apply(pd.to_numeric, errors='coerce')  
    return df