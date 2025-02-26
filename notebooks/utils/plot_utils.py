from .recbole_train_test import get_test_data_sections, get_evaluation_results_filename
from .generate_artificial_random_dataset import load_picklefile


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
import plotly.offline as py

pd.options.plotting.backend = "plotly"
py.init_notebook_mode() # graphs charts inline (IPython).

a4_dims = (11.7, 8.27)


def recall_heatmap(df,
                   round_point=2,
                   title='Recall@20 for checkpoint models across Holdouts - model - data',
                   filepath='images/heatmaps/..'):
    
    # code from Measuring Forgetting  
    plt.figure(figsize=(15, 10))
    x_t = np.arange(0, df.shape[0])
    labels=[str(i+1) for i in x_t]
    sns.heatmap(df, vmin=0, vmax=df.max().max(), annot=True, fmt=f'0.{round_point}f', linewidths=.1, cmap='Spectral_r', xticklabels=labels, yticklabels=labels)
    plt.ylabel('model')
    plt.xlabel('holdout')
    plt.title(title)
    if filepath:
        plt.savefig(filepath);
    else:
        plt.show()



def get_results_matrix(model_name,
                       models_versions, 
                       base_dataset_name, 
                       save_path, 
                       metric='recall@3', 
                       filename_version='',
                       part_shift_incl=True):
    '''
        model_versions: list ['_pt1', '_pt2', '_pt3', '']

        base_dataset_name: str
        save_path: str
        metric: str key of the ordered dictionary with the test results 
                example: OrderedDict([('recall@3', 1.0),
                                        ('mrr@3', 0.818),
                                        ('ndcg@3', 0.8657),
                                        ('hit@3', 1.0),
                                        ('precision@3', 0.3333)])


                Dataset section parts
                    |    part 8    |
            |part 1|part 5| part 6|part 7|
            |    part 2   |
            |         part 3      |
            |             ''             |


        columns (sections)
            0: pt1
            1: pt5
            2: pt6
            3: pt7
            4: pt8

        index (model)
            0: pt1
            1: pt2
            2: pt3
            3: '' (full, pt4)

    '''

    cols = [i for i in range(len(models_versions))]
    df = pd.DataFrame(columns=cols) 

    for row_idx, model_ver in enumerate(models_versions):
        # print('\n\n', row_idx, model_ver)
        print('\n\n',model_ver)
        current_dataset_name = base_dataset_name+model_ver


        # test_results_diagonal = load_picklefile(save_path+base_dataset_name+'/test_results_diagonal.pkl')
        test_recall_diagonal = load_picklefile(save_path+current_dataset_name+'/'+\
                                    get_evaluation_results_filename(model_name,
                                                                    None,
                                                                    'diagonal',
                                                                    filename_version)+'.pkl')[metric]
        df.loc[row_idx,row_idx] = test_recall_diagonal


        
        test_data_sections = get_test_data_sections(model_version=model_ver,
                                                    models_versions=models_versions) 
        test_data_sections = test_data_sections if part_shift_incl else test_data_sections[:-1]

        
        # triangles
        # print(test_data_sections)
        for column_idx, section in enumerate(test_data_sections):
            test_recall = load_picklefile(save_path+current_dataset_name+\
                                '/'+get_evaluation_results_filename(model_name,
                                                                    current_dataset_name,
                                                                    section,
                                                                    filename_version)+'.pkl')[metric]

            # print('row_idx='+str(row_idx)+'; column_idx='+str(column_idx))
            if column_idx < row_idx:
                # print('column_idx < row_idx -> row_idx='+str(row_idx)+'; column_idx='+str(column_idx))
                # print('column_idx='+str(column_idx)+'; row_idx='+str(row_idx))
                df.loc[row_idx,column_idx] = test_recall
            else:
                # print('column_idx  >= row_idx -> row_idx='+str(row_idx)+'; column_idx='+str(column_idx+1))
                # column_idx >= row_idx:
                # print('column_idx+1='+str(column_idx+1)+'; row_idx='+str(row_idx))
                df.loc[row_idx,column_idx+1] = test_recall
        
        # print(df)

    df = df.apply(pd.to_numeric, errors='coerce')  
    return df

# def get_results_matrix(model_name,
#                        model_versions, 
#                        base_dataset_name, 
#                        save_path, 
#                        metric='recall@3', 
#                        filename_version='',
#                        pt8=True):
#     '''
#         model_versions: list ['_pt1', '_pt2', '_pt3', '']

#         base_dataset_name: str
#         save_path: str
#         metric: str key of the ordered dictionary with the test results 
#                 example: OrderedDict([('recall@3', 1.0),
#                                         ('mrr@3', 0.818),
#                                         ('ndcg@3', 0.8657),
#                                         ('hit@3', 1.0),
#                                         ('precision@3', 0.3333)])


#                 Dataset section parts
#                     |    part 8    |
#             |part 1|part 5| part 6|part 7|
#             |    part 2   |
#             |         part 3      |
#             |             ''             |


#         columns (sections)
#             0: pt1
#             1: pt5
#             2: pt6
#             3: pt7
#             4: pt8

#         index (model)
#             0: pt1
#             1: pt2
#             2: pt3
#             3: '' (full, pt4)

#     '''

#     df = pd.DataFrame(columns=[0,1,2,3,4]) 

#     for row_idx, model_ver in enumerate(model_versions):
#         print('\n\n'+model_ver)
#         current_dataset_name = base_dataset_name+model_ver


#         # test_results_diagonal = load_picklefile(save_path+base_dataset_name+'/test_results_diagonal.pkl')
#         test_recall_diagonal = load_picklefile(save_path+current_dataset_name+'/'+\
#                                                 get_evaluation_results_filename(model_name,
#                                                                                 None,
#                                                                                 'diagonal',
#                                                                                 filename_version)+'.pkl')\
#                                     [metric]
#         df.loc[row_idx,row_idx] = test_recall_diagonal


        
#         test_data_sections = get_test_data_sections(model_version=model_ver) 
#         test_data_sections = test_data_sections if pt8 else test_data_sections[:-1]

        
#         # print(test_data_sections)
#         for column_idx, section in enumerate(test_data_sections):
#             test_recall = load_picklefile(save_path+current_dataset_name+\
#                                             '/'+get_evaluation_results_filename(model_name,
#                                                                                 current_dataset_name,
#                                                                                 section,
#                                                                                 filename_version)+'.pkl')[metric]

#             if column_idx < row_idx:
#                 # print('column_idx='+str(column_idx)+'; row_idx='+str(row_idx))
#                 df.loc[row_idx,column_idx] = test_recall
#             else:
#                 # column_idx >= row_idx:
#                 # print('column_idx+1='+str(column_idx+1)+'; row_idx='+str(row_idx))
#                 df.loc[row_idx,column_idx+1] = test_recall

#     df = df.apply(pd.to_numeric, errors='coerce')  
#     return df