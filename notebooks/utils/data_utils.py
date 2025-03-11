
import pandas as pd
from .generate_artificial_random_dataset import validate_folderpath


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
