import pandas as pd
from .file_utils import validate_folderpath

from .train_evaluate import get_folderpath_str


def get_id2token(recbole_dataset, col):
    if col=='item_id':
        return recbole_dataset.id2token(recbole_dataset.iid_field, recbole_dataset.inter_feat.interaction[recbole_dataset.iid_field])
    elif col=='user_id':
        return recbole_dataset.id2token(recbole_dataset.uid_field, recbole_dataset.inter_feat.interaction[recbole_dataset.uid_field])
    return None


def recbole_ds_column_2_dataframe(recbole_dataset, col, data_type):
    recboleds_df = pd.DataFrame(get_id2token(recbole_dataset, col=col), columns=[col])
    recboleds_df[col] = recboleds_df[col].astype(data_type)
    return recboleds_df
        

def recbole_dataset_2_external_id_df(recbole_dataset, data_types_dict):
    i = recbole_ds_column_2_dataframe(recbole_dataset, col=recbole_dataset.iid_field, data_type=data_types_dict[recbole_dataset.iid_field])
    u = recbole_ds_column_2_dataframe(recbole_dataset, col=recbole_dataset.uid_field, data_type=data_types_dict[recbole_dataset.uid_field])

    df = pd.DataFrame({'user_id': u.user_id,
                       'item_id': i.item_id,
                       'timestamp': recbole_dataset.inter_feat.interaction[recbole_dataset.time_field].numpy()})
    df['timestamp'] = df['timestamp'].astype(data_types_dict['timestamp'])

    return df


def get_all_users_all_items_initialised(df, time_col='timestamp'):

    df_users = pd.DataFrame()
    df_users['user_id'] = list(df.user_id.unique())
    df_users['item_id'] = 'i0'

    df_items = pd.DataFrame()
    df_items['item_id'] = list(df.item_id.unique())
    df_items['user_id'] = 'u0'
    
    users_items_init = pd.concat([df_users, df_items]) 
    users_items_init[time_col] = df[time_col].min()

    return users_items_init


def calculate_sparsity(df):
        # df.item_id.groupby([df.user_id, df.item_id]).count().sum() == df.user_id.count()
        sparsity = 1 - df.user_id.count()/(df.user_id.nunique()*df.item_id.nunique())
        specs_str = str(df.user_id.nunique())+'x'+str(df.item_id.nunique())+'_'+str(round(sparsity, 2))
        print('specs_str', specs_str)
        return sparsity, specs_str
    

# def rename_item(row):
#     global sudden_drift_start
#     global renamed_items
    
#     # print('sudden_drift_start=', sudden_drift_start)
#     # print('renamed_items=', renamed_items)

#     if int(row.name) > sudden_drift_start and row['item_id'] in renamed_items:
#         return renamed_items[row['item_id']]
#     return row['item_id']


def save_dataframe_2_atomic_file(df, save_path, base_filename, specs_str, benchmark_filename):
    if save_path:
        folderpath = get_folderpath_str(save_path, base_filename, specs_str)
        validate_folderpath(folderpath)
        # print(folderpath)
        # Output the dataset
        filename = base_filename+'_'+specs_str+'.'+benchmark_filename
        filepath = folderpath+filename
        # print(filepath)
        df = df[['user_id', 'item_id', 'timestamp']]
        # print(df.head())


        df.to_csv(filepath+'.csv', index=False)
        df.to_csv(filepath+'.inter',
                            header=['user_id:token','item_id:token','timestamp:float'], 
                            sep='\t', 
                            index=False)
        print("Dataset saved at "+folderpath+", named "+filename+", in .csv and .inter")


def save_complete_dataset_atomic_file(df, save_path, base_filename, specs_str):
    if save_path:
        folderpath = get_folderpath_str(save_path, base_filename, specs_str)
        validate_folderpath(folderpath)
        # print(folderpath)
        # Output the dataset
        filepath = folderpath+base_filename+'_'+specs_str
        # print(filepath)
        df = df[['user_id', 'item_id', 'timestamp']]
        # print(df.head())

        df.to_csv(filepath+'.csv', index=False)
        df.to_csv(filepath+'.inter',
                            header=['user_id:token','item_id:token','timestamp:float'], 
                            sep='\t', 
                            index=False)
        print("Dataset with sudden drift created and saved at "+filepath+".")
        print(df.item_id.value_counts())